# ============================================================================
# 03_evaluation_fixed_opt.R - EVALUACIÓN ROBUSTA (BASE + ENSEMBLES, OPTIMIZADA)
# ============================================================================

library(nnet)
library(e1071)
library(randomForest)

# ----------------------------------------------------------------------------
# Cálculo de métricas (robusto a NaNs / clases sin soporte)
# ----------------------------------------------------------------------------
compute_confusion_metrics <- function(y_true, y_pred) {
  # Forzar que ambos sean factores con los mismos niveles
  all_levels <- sort(union(levels(y_true), levels(y_pred)))
  y_true <- factor(y_true, levels = all_levels)
  y_pred <- factor(y_pred, levels = all_levels)
  
  conf_matrix <- table(y_true, y_pred)
  diag_vals <- diag(conf_matrix)
  total_preds <- sum(conf_matrix)
  
  if (total_preds == 0) {
    return(list(accuracy = 0, f1 = 0))
  }
  
  # Precisión
  col_sums  <- colSums(conf_matrix)
  precision <- ifelse(col_sums == 0, 0, diag_vals / col_sums)
  
  # Recall
  row_sums <- rowSums(conf_matrix)
  recall   <- ifelse(row_sums == 0, 0, diag_vals / row_sums)
  
  # F1 por clase
  f1_per_class <- ifelse(
    (precision + recall) == 0,
    0,
    2 * (precision * recall) / (precision + recall)
  )
  
  macro_f1   <- mean(f1_per_class, na.rm = TRUE)
  overall_acc <- sum(diag_vals) / total_preds
  
  list(accuracy = overall_acc, f1 = macro_f1)
}

# ----------------------------------------------------------------------------
# Probabilidades alineadas (n_samples x length(levels_order)) o NULL
# ----------------------------------------------------------------------------
get_probabilities_aligned <- function(model, X, levels_order) {
  X_df      <- as.data.frame(X)
  n_rows    <- nrow(X_df)
  n_classes <- length(levels_order)
  
  raw_probs <- tryCatch({
    if (inherits(model, "nnet")) {
      p <- predict(model, X_df, type = "raw")
      if (is.vector(p)) {
        p <- matrix(p, nrow = n_rows, byrow = TRUE)
      }
      p
    } else if (inherits(model, "svm")) {
      attr(predict(model, X_df, probability = TRUE), "probabilities")
    } else if (inherits(model, "randomForest")) {
      predict(model, X_df, type = "prob")
    } else {
      return(NULL)
    }
  }, error = function(e) {
    warning(sprintf("get_probabilities_aligned: predict() falló: %s", e$message))
    NULL
  })
  
  if (is.null(raw_probs)) return(NULL)
  
  probs <- tryCatch(as.matrix(raw_probs), error = function(e) NULL)
  if (is.null(probs)) return(NULL)
  
  # Normalizar dimensiones
  if (is.null(dim(probs))) {
    if (length(probs) == n_rows) {
      probs <- matrix(probs, nrow = n_rows, ncol = 1)
    } else {
      warning(sprintf(
        "get_probabilities_aligned: vector length %d incompatible con n_rows = %d",
        length(probs), n_rows
      ))
      return(NULL)
    }
  }
  
  if (nrow(probs) != n_rows) {
    warning(sprintf(
      "get_probabilities_aligned: nrow(probs) = %d != n_rows = %d",
      nrow(probs), n_rows
    ))
    return(NULL)
  }
  
  probs[is.na(probs)] <- 0
  
  # Alinear con levels_order
  aligned <- matrix(0, nrow = n_rows, ncol = n_classes)
  colnames(aligned) <- levels_order
  
  if (!is.null(colnames(probs))) {
    common <- intersect(colnames(probs), levels_order)
    if (length(common) > 0) {
      aligned[, common] <- probs[, common, drop = FALSE]
      return(aligned)
    }
    if (ncol(probs) == n_classes) {
      aligned[] <- probs
      return(aligned)
    }
    warning("get_probabilities_aligned: colnames(probs) no coinciden con levels_order")
    return(NULL)
  } else {
    # Sin nombres: si el nº de columnas coincide, asumimos orden de levels_order
    if (ncol(probs) == n_classes) {
      aligned[] <- probs
      return(aligned)
    }
    warning(sprintf(
      "get_probabilities_aligned: ncol(probs) = %d != n_classes = %d y sin nombres",
      ncol(probs), n_classes
    ))
    return(NULL)
  }
}

# ----------------------------------------------------------------------------
# Construcción de meta-features para un conjunto de modelos base
# (se llama UNA sola vez por dataset y se reutiliza en todos los ensembles)
# ----------------------------------------------------------------------------
build_meta_features_once <- function(models, base_names, X, levels_order) {
  n_samples <- nrow(X)
  n_classes <- length(levels_order)
  n_base    <- length(base_names)
  
  if (n_base == 0) {
    warning("build_meta_features_once: no hay modelos base válidos")
    return(NULL)
  }
  
  meta_features <- matrix(0, nrow = n_samples, ncol = n_base * n_classes)
  
  for (i in seq_along(base_names)) {
    nm <- base_names[[i]]
    model_wrapper <- models[[nm]]
    if (is.null(model_wrapper$model)) next
    
    probs <- get_probabilities_aligned(model_wrapper$model, X, levels_order)
    if (is.null(probs)) {
      warning(sprintf("build_meta_features_once: modelo base '%s' sin probs válidas", nm))
      next
    }
    
    col_start <- (i - 1) * n_classes + 1
    col_end   <- i * n_classes
    meta_features[, col_start:col_end] <- probs
  }
  
  meta_features[is.na(meta_features)] <- 0
  meta_features
}

# ----------------------------------------------------------------------------
# Predicción de etiquetas (base o ensemble)
# ----------------------------------------------------------------------------
predict_labels <- function(model, X, levels_order) {
  tryCatch({
    raw_pred <- predict(model, X)
    
    # Caso MLP ensemble con salida softmax (matriz)
    if (inherits(model, "nnet") && is.matrix(raw_pred)) {
      max_idx <- max.col(raw_pred, ties.method = "first")
      if (!is.null(colnames(raw_pred))) {
        pred_class <- colnames(raw_pred)[max_idx]
      } else {
        pred_class <- levels_order[max_idx]
      }
      return(factor(pred_class, levels = levels_order))
    }
    
    # Caso estándar (randomForest, svm, etc.)
    factor(raw_pred, levels = levels_order)
  }, error = function(e) {
    warning(paste("Error en predict_labels:", e$message))
    factor(rep(levels_order[1], nrow(X)), levels = levels_order)
  })
}

# ----------------------------------------------------------------------------
# Evaluación de todos los modelos de un conjunto (base + ensembles)
# ----------------------------------------------------------------------------
evaluate_models <- function(models, dataset_name, X_test, y_test) {
  cat(sprintf("\n== Evaluando modelos para [%s] ==\n", toupper(dataset_name)))
  
  levels_order <- levels(y_test)
  if (is.null(levels_order) || length(levels_order) == 0) {
    stop("y_test debe ser un factor con niveles definidos.")
  }
  
  # Identificar ensembles y modelos base
  types <- sapply(models, function(m) if (!is.null(m$type)) m$type else "unknown")
  ensemble_names <- names(models)[grepl("^ensemble_", types)]
  
  # Determinar modelos base a partir del PRIMER ensemble (asumimos todos iguales)
  base_names <- NULL
  if (length(ensemble_names) > 0) {
    first_ens <- models[[ensemble_names[1]]]
    if (!is.null(first_ens$base_model_names)) {
      base_names <- first_ens$base_model_names
    } else {
      base_names <- names(models)[
        sapply(models, function(m) !grepl("^ensemble_", m$type) && !is.null(m$model))
      ]
    }
  }
  
  # Construir meta-features UNA sola vez si hay ensembles
  meta_X <- NULL
  if (length(ensemble_names) > 0) {
    cat("   Construyendo meta-features (una vez) para ensembles...\n")
    meta_X <- build_meta_features_once(models, base_names, X_test, levels_order)
    if (is.null(meta_X)) {
      warning("No se pudieron construir meta-features; se saltarán ensembles.")
      ensemble_names <- character(0)
    }
  }
  
  results_list <- list()
  
  for (model_name in names(models)) {
    info <- models[[model_name]]
    if (is.null(info$model)) next
    
    model <- info$model
    type  <- if (!is.null(info$type)) info$type else "unknown"
    
    cat(sprintf("-> Evaluando: %-20s (%s)... ", model_name, type))
    
    # 1) Elegir X_eval
    if (model_name %in% ensemble_names) {
      if (is.null(meta_X)) {
        cat("SIN META-FEATURES\n")
        next
      }
      X_eval <- meta_X
    } else {
      X_eval <- X_test
    }
    
    # 2) Predicción + métricas
    start   <- Sys.time()
    pred    <- predict_labels(model, X_eval, levels_order)
    elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
    
    metrics <- compute_confusion_metrics(y_test, pred)
    
    results_list[[model_name]] <- data.frame(
      dataset  = dataset_name,
      model    = model_name,
      type     = type,
      accuracy = metrics$accuracy,
      f1       = metrics$f1,
      time_sec = elapsed,
      stringsAsFactors = FALSE
    )
    
    cat(sprintf("Acc: %.4f | F1: %.4f\n", metrics$accuracy, metrics$f1))
  }
  
  if (length(results_list) == 0) return(NULL)
  do.call(rbind, results_list)
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
main <- function() {
  if (!dir.exists("evaluation")) dir.create("evaluation", recursive = TRUE)
  
  # ---------------- PCA ----------------
  cat("\n=== EVALUANDO PCA ===\n")
  load("processed_data/mnist_pca.RData")  # pca_result, train_labels, test_labels
  
  X_test_pca <- as.data.frame(pca_result$test)
  y_test_pca <- factor(test_labels, levels = sort(unique(train_labels)))
  rm(pca_result); gc()
  
  load("trained_models/models_pca.RData")  # models_pca
  res_pca <- evaluate_models(models_pca, "pca", X_test_pca, y_test_pca)
  rm(models_pca); gc()
  
  # ---------------- CROP ----------------
  cat("\n=== EVALUANDO CROP ===\n")
  load("processed_data/mnist_crop_pooling.RData")  # crop_result, train_labels, test_labels
  
  X_test_crop <- as.data.frame(crop_result$test)
  y_test_crop <- factor(test_labels, levels = sort(unique(train_labels)))
  rm(crop_result); gc()
  
  load("trained_models/models_crop_pooling.RData")  # models_crop
  res_crop <- evaluate_models(models_crop, "crop", X_test_crop, y_test_crop)
  rm(models_crop); gc()
  
  # ---------------- Resultados finales ----------------
  final_results <- rbind(res_pca, res_crop)
  print(final_results)
  write.csv(final_results, "evaluation/test_metrics.csv", row.names = FALSE)
  
  cat("\nResultados guardados en 'evaluation/test_metrics.csv'\n")
}

main()
