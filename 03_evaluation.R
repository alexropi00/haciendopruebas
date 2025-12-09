# ============================================================================
# 03_evaluation_fixed.R - EVALUACIÓN CON MANEJO DE ERRORES DE ENSEMBLE
# ============================================================================

library(nnet)
library(e1071)
library(randomForest)

# ----------------------------------------------------------------------------
# Cálculo de métricas (Protegido contra NaNs)
# ----------------------------------------------------------------------------
compute_confusion_metrics <- function(y_true, y_pred) {
  # Forzar que ambos sean factores con los mismos niveles
  all_levels <- sort(union(levels(y_true), levels(y_pred)))
  y_true <- factor(y_true, levels = all_levels)
  y_pred <- factor(y_pred, levels = all_levels)
  
  conf_matrix <- table(y_true, y_pred)
  diag_vals <- diag(conf_matrix)
  total_preds <- sum(conf_matrix)
  
  if (total_preds == 0) return(list(accuracy = 0, f1 = 0))
  
  # Precisión
  col_sums <- colSums(conf_matrix)
  precision <- ifelse(col_sums == 0, 0, diag_vals / col_sums)
  
  # Recall
  row_sums <- rowSums(conf_matrix)
  recall <- ifelse(row_sums == 0, 0, diag_vals / row_sums)
  
  # F1
  f1_per_class <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
  
  macro_f1 <- mean(f1_per_class, na.rm = TRUE)
  overall_acc <- sum(diag_vals) / total_preds
  
  list(accuracy = overall_acc, f1 = macro_f1)
}

# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
get_probabilities <- function(model, X) {
  tryCatch({
    if (inherits(model, "nnet")) {
      probs <- predict(model, X, type = "raw")
      if (is.vector(probs)) probs <- matrix(probs, nrow = nrow(X), byrow = TRUE)
      return(probs)
    }
    if (inherits(model, "svm")) {
      attr(predict(model, X, probability = TRUE), "probabilities")
    } else if (inherits(model, "randomForest")) {
      predict(model, X, type = "prob")
    } else {
      NULL
    }
  }, error = function(e) NULL)
}

build_meta_features <- function(ensemble_info, all_models, X, levels_order) {
  # Filtro estricto: solo modelos base válidos
  is_base <- sapply(all_models, function(m) !grepl("^ensemble_", m$type) && !is.null(m$model))
  base_models <- all_models[is_base]
  
  if (length(base_models) == 0) return(NULL)
  
  # Matriz vacía
  n_base <- length(base_models)
  meta_features <- matrix(0, nrow = nrow(X), ncol = n_base * length(levels_order))
  
  for (i in seq_along(base_models)) {
    model_wrapper <- base_models[[i]]
    probs <- get_probabilities(model_wrapper$model, X)
    
    if (!is.null(probs)) {
      # Asegurar que las columnas coincidan con los niveles esperados
      # Si faltan columnas, rellenar con 0
      aligned_probs <- matrix(0, nrow = nrow(X), ncol = length(levels_order), 
                              dimnames = list(NULL, levels_order))
      
      present_cols <- intersect(colnames(probs), levels_order)
      if (length(present_cols) > 0) {
        aligned_probs[, present_cols] <- probs[, present_cols]
      }
      
      col_start <- (i - 1) * length(levels_order) + 1
      col_end <- i * length(levels_order)
      meta_features[, col_start:col_end] <- aligned_probs
    }
    gc()
  }
  
  # Reemplazar NAs por 0 para que SVM/RF no fallen
  meta_features[is.na(meta_features)] <- 0
  return(meta_features)
}

predict_labels <- function(model, X, levels_order) {
  tryCatch({
    raw_pred <- predict(model, X)
    
    # Manejo específico para NNET (Softmax output)
    if (inherits(model, "nnet") && is.matrix(raw_pred)) {
      max_idx <- max.col(raw_pred, ties.method = "first")
      # Mapear índice a nombre de columna si existe, o usar levels_order
      if (!is.null(colnames(raw_pred))) {
        pred_class <- colnames(raw_pred)[max_idx]
      } else {
        pred_class <- levels_order[max_idx]
      }
      return(factor(pred_class, levels = levels_order))
    }
    
    # Manejo estándar
    factor(raw_pred, levels = levels_order)
  }, error = function(e) {
    # Fallback en caso de error catastrófico: devolver primera clase
    warning(paste("Error predicción:", e$message))
    factor(rep(levels_order[1], nrow(X)), levels = levels_order)
  })
}

# ----------------------------------------------------------------------------
# Evaluación
# ----------------------------------------------------------------------------
evaluate_models <- function(models, dataset_name, X_test, y_test) {
  cat(sprintf("\n== Evaluando modelos para [%s] ==\n", toupper(dataset_name)))
  levels_order <- levels(y_test)
  results_list <- list()
  
  for (model_name in names(models)) {
    info <- models[[model_name]]
    if (is.null(info$model)) next
    
    model <- info$model
    cat(sprintf("-> Evaluando: %-20s (%s)... ", model_name, info$type))
    
    X_eval <- X_test
    
    if (grepl("^ensemble_", info$type)) {
      X_eval <- build_meta_features(info, models, X_test, levels_order)
      if (is.null(X_eval)) {
        cat("ERROR META-FEATURES\n")
        next
      }
    }
    
    start <- Sys.time()
    pred <- predict_labels(model, X_eval, levels_order)
    elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
    
    metrics <- compute_confusion_metrics(y_test, pred)
    
    results_list[[model_name]] <- data.frame(
      dataset = dataset_name,
      model = model_name,
      type = info$type,
      accuracy = metrics$accuracy,
      f1 = metrics$f1,
      time_sec = elapsed,
      stringsAsFactors = FALSE
    )
    
    cat(sprintf("Acc: %.4f | F1: %.4f\n", metrics$accuracy, metrics$f1))
    gc()
  }
  
  if (length(results_list) == 0) return(NULL)
  do.call(rbind, results_list)
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
main <- function() {
  if (!dir.exists("evaluation")) dir.create("evaluation")
  
  # Cargar datos
  load("processed_data/mnist_pca.RData")
  load("processed_data/mnist_crop_pooling.RData")
  
  # Evaluar PCA
  load("trained_models/models_pca.RData")
  y_test_pca <- factor(test_labels, levels = sort(unique(train_labels)))
  res_pca <- evaluate_models(models_pca, "pca", pca_result$test, y_test_pca)
  rm(models_pca); gc()
  
  # Evaluar CROP
  load("trained_models/models_crop_pooling.RData")
  y_test_crop <- factor(test_labels, levels = sort(unique(train_labels)))
  res_crop <- evaluate_models(models_crop, "crop", crop_result$test, y_test_crop)
  rm(models_crop); gc()
  
  final_results <- rbind(res_pca, res_crop)
  print(final_results)
  write.csv(final_results, "evaluation/test_metrics.csv", row.names = FALSE)
}

main()
