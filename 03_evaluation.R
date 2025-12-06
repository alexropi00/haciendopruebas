# ============================================================================
# 03_evaluation.R - Evaluación de modelos entrenados sobre el conjunto de test
# ============================================================================
# Este script carga los modelos generados en 02_training.R y los evalúa sobre
# el conjunto de test de cada variante de preprocesamiento (PCA y Crop+Pooling).
# Para cada modelo se calculan métricas clave (accuracy, loss, F1-score) y se
# mide el tiempo total necesario para procesar el conjunto de test completo.
# ============================================================================

library(nnet)
library(e1071)
library(randomForest)

# ----------------------------------------------------------------------------
# Cálculo de métricas
# ----------------------------------------------------------------------------
compute_confusion_metrics <- function(y_true, y_pred) {
  # Asegurar que no haya predicciones NA para evitar divisiones 0/0.
  if (any(is.na(y_pred))) {
    fill_label <- levels(y_true)[1]
    y_pred <- factor(ifelse(is.na(y_pred), fill_label, as.character(y_pred)), levels = levels(y_true))
  }

  tbl <- table(y_true, y_pred)
  classes <- union(levels(y_true), levels(y_pred))
  if (!all(classes %in% rownames(tbl))) {
    missing_rows <- setdiff(classes, rownames(tbl))
    for (cls in missing_rows) tbl <- rbind(tbl, setNames(rep(0, ncol(tbl)), colnames(tbl)))
    rownames(tbl) <- classes
  }
  if (!all(classes %in% colnames(tbl))) {
    missing_cols <- setdiff(classes, colnames(tbl))
    for (cls in missing_cols) tbl <- cbind(tbl, setNames(rep(0, nrow(tbl)), cls))
    colnames(tbl) <- classes
  }
  diag_vals <- diag(tbl)
  totals_pred <- colSums(tbl)
  totals_true <- rowSums(tbl)

  precision <- ifelse(totals_pred == 0, 0, diag_vals / totals_pred)
  recall <- ifelse(totals_true == 0, 0, diag_vals / totals_true)
  f1_per_class <- ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))

  macro_f1 <- mean(f1_per_class)
  overall_acc <- sum(diag_vals) / sum(tbl)

  list(accuracy = overall_acc, f1 = macro_f1)
}

compute_log_loss <- function(probs, y_true) {
  probs[is.na(probs)] <- 0

  # Asegurar que las columnas del arreglo de probabilidades estén alineadas con
  # los niveles de las etiquetas.
  levels_order <- levels(y_true)
  if (is.null(colnames(probs))) {
    if (ncol(probs) == length(levels_order)) {
      colnames(probs) <- levels_order
    } else {
      warning(
        sprintf(
          "No se pueden alinear las probabilidades: se esperaban %d columnas y se recibieron %d.",
          length(levels_order), ncol(probs)
        )
      )
      return(NA_real_)
    }
  }

  missing_cols <- setdiff(levels_order, colnames(probs))
  if (length(missing_cols) > 0) {
    warning(
      sprintf(
        "Faltan probabilidades para algunas clases en la predicción: %s",
        paste(missing_cols, collapse = ", ")
      )
    )
    return(NA_real_)
  }

  probs_aligned <- probs[, levels_order, drop = FALSE]
  probs_clipped <- pmin(pmax(probs_aligned, 1e-15), 1 - 1e-15)

  col_index <- match(y_true, levels_order)
  if (any(is.na(col_index))) {
    warning("No se pudieron alinear las etiquetas verdaderas con las columnas de probabilidad")
    return(NA_real_)
  }

  true_indices <- cbind(seq_along(y_true), col_index)
  true_probs <- probs_clipped[true_indices]

  -mean(log(true_probs))
}

# ----------------------------------------------------------------------------
# Utilidades de predicción
# ----------------------------------------------------------------------------
get_probabilities <- function(model, X) {
  if (inherits(model, "nnet")) {
    p <- predict(model, X, type = "raw")
    if (is.vector(p)) p <- matrix(p, nrow = nrow(X), byrow = TRUE)
    p[is.na(p)] <- 0
    return(p)
  }

  if (inherits(model, "svm")) {
    p <- attr(predict(model, X, probability = TRUE), "probabilities")
    p[is.na(p)] <- 0
    return(p)
  }

  if (inherits(model, "randomForest")) {
    if (!is.null(model$type) && model$type == "classification") {
      p <- tryCatch(predict(model, X, type = "prob"), error = function(e) NULL)
      if (!is.null(p)) {
        p <- as.matrix(p)
        p[is.na(p)] <- 0
        return(p)
      }
    }
    return(NULL)
  }

  NULL
}

prepare_prob_matrix <- function(probs, levels_order, n_rows) {
  if (is.null(probs)) {
    return(matrix(0, nrow = n_rows, ncol = length(levels_order)))
  }

  if (is.null(colnames(probs)) && ncol(probs) == length(levels_order)) {
    colnames(probs) <- levels_order
  }

  missing_cols <- setdiff(levels_order, colnames(probs))
  if (length(missing_cols) > 0 || ncol(probs) != length(levels_order)) {
    warning(
      sprintf(
        "No se pudieron alinear las probabilidades para el meta-modelo: se esperaban %d columnas y se recibieron %d.",
        length(levels_order), ncol(probs)
      )
    )
    return(matrix(0, nrow = nrow(probs), ncol = length(levels_order)))
  }

  probs[, levels_order, drop = FALSE]
}

build_meta_features <- function(models, X, levels_order, max_base_models = 5) {
  base_idx <- which(!grepl("^ensemble_", vapply(models, function(m) m$type, character(1))) &
                      vapply(models, function(m) isTRUE(m$success) || is.null(m$success), logical(1)))
  if (length(base_idx) == 0) {
    return(NULL)
  }

  selected_idx <- head(base_idx, max_base_models)
  selected <- models[selected_idx]

  meta <- matrix(0, nrow = nrow(X), ncol = length(selected) * length(levels_order))
  for (i in seq_along(selected)) {
    probs <- get_probabilities(selected[[i]]$model, X)
    aligned <- prepare_prob_matrix(probs, levels_order, nrow(X))
    col_range <- ((i - 1) * length(levels_order) + 1):(i * length(levels_order))
    meta[, col_range] <- aligned
  }

  meta
}

predict_labels <- function(model, X, levels_order) {
  probs <- get_probabilities(model, X)
  if (!is.null(probs)) {
    if (is.null(colnames(probs)) && ncol(probs) == length(levels_order)) {
      colnames(probs) <- levels_order
    }

    aligned_probs <- probs
    if (!is.null(colnames(probs)) && all(levels_order %in% colnames(probs))) {
      aligned_probs <- probs[, levels_order, drop = FALSE]
    }

    preds <- apply(aligned_probs, 1, function(r) {
      if (all(is.na(r))) return(NA_character_)
      idx <- which.max(r)
      lvl <- if (!is.null(colnames(aligned_probs))) colnames(aligned_probs)[idx] else idx - 1
      lvl
    })

    labels <- factor(preds, levels = levels_order)
    if (any(is.na(labels))) {
      raw_pred <- predict(model, X)
      labels <- factor(raw_pred, levels = levels_order)
      return(list(labels = labels, probs = NULL))
    }

    return(list(labels = labels, probs = aligned_probs))
  }

  raw_pred <- predict(model, X)
  list(labels = factor(raw_pred, levels = levels_order), probs = NULL)
}

# ----------------------------------------------------------------------------
# Evaluación de un conjunto de modelos sobre un dataset concreto
# ----------------------------------------------------------------------------
evaluate_models <- function(models, dataset_name, X_test, y_test) {
  cat(sprintf("\n== Evaluando modelos (%s) ==\n", dataset_name))
  levels_order <- levels(y_test)

  results <- lapply(names(models), function(model_name) {
    info <- models[[model_name]]
    model <- info$model

    X_eval <- X_test
    if (grepl("^ensemble_", info$type)) {
      X_meta <- build_meta_features(models, X_test, levels_order)
      if (is.null(X_meta)) {
        warning(sprintf("No se pudieron generar características meta para el modelo %s; se omite.", model_name))
        return(NULL)
      }
      X_eval <- X_meta
    }

    start_time <- Sys.time()
    pred_data <- predict_labels(model, X_eval, levels_order)
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

    metrics <- compute_confusion_metrics(y_test, pred_data$labels)
    loss <- NA_real_
    if (!is.null(pred_data$probs)) {
      loss <- compute_log_loss(pred_data$probs, y_test)
    }

    list(
      dataset = dataset_name,
      model = model_name,
      type = info$type,
      accuracy = metrics$accuracy,
      f1 = metrics$f1,
      loss = loss,
      inference_time_sec = elapsed
    )
  })

  results <- Filter(Negate(is.null), results)
  if (length(results) == 0) return(data.frame())

  do.call(rbind, lapply(results, as.data.frame))
}

# ----------------------------------------------------------------------------
# Carga de datos y ejecución principal
# ----------------------------------------------------------------------------
main <- function() {
  if (!file.exists("trained_models/models_pca.RData") ||
      !file.exists("trained_models/models_crop_pooling.RData")) {
    stop("No se encontraron los archivos de modelos en 'trained_models/'. Ejecuta 02_training.R primero.")
  }

  cat("Cargando datos procesados...\n")
  load("processed_data/mnist_pca.RData")
  load("processed_data/mnist_crop_pooling.RData")

  cat("Cargando modelos entrenados...\n")
  load("trained_models/models_pca.RData")
  load("trained_models/models_crop_pooling.RData")

  y_test_pca <- factor(test_labels, levels = sort(unique(train_labels)))
  y_test_crop <- factor(test_labels, levels = sort(unique(train_labels)))

  results_pca <- evaluate_models(models_pca, "pca", pca_result$test, y_test_pca)
  results_crop <- evaluate_models(models_crop, "crop_pooling", crop_result$test, y_test_crop)

  results <- rbind(results_pca, results_crop)
  print(results)

  if (!dir.exists("evaluation")) dir.create("evaluation")
  write.csv(results, file = "evaluation/test_metrics.csv", row.names = FALSE)
  cat("\nResultados guardados en evaluation/test_metrics.csv\n")
}

main()
