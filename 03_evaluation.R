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
  # Asegurar que las columnas del arreglo de probabilidades estén alineadas con
  # los niveles de las etiquetas.
  levels_order <- levels(y_true)
  if (is.null(colnames(probs))) {
    if (ncol(probs) != length(levels_order)) {
      stop("No se pueden alinear las probabilidades sin nombres de columna válidos")
    }
    colnames(probs) <- levels_order
  }

  missing_cols <- setdiff(levels_order, colnames(probs))
  if (length(missing_cols) > 0) {
    stop("Faltan probabilidades para algunas clases en la predicción: ", paste(missing_cols, collapse = ", "))
  }

  probs_aligned <- probs[, levels_order, drop = FALSE]
  probs_clipped <- pmin(pmax(probs_aligned, 1e-15), 1 - 1e-15)

  true_indices <- cbind(seq_along(y_true), as.character(y_true))
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
    return(p)
  }

  if (inherits(model, "svm")) {
    p <- attr(predict(model, X, probability = TRUE), "probabilities")
    return(p)
  }

  if (inherits(model, "randomForest")) {
    p <- predict(model, X, type = "prob")
    return(as.matrix(p))
  }

  NULL
}

predict_labels <- function(model, X, levels_order) {
  probs <- get_probabilities(model, X)
  if (!is.null(probs)) {
    if (is.null(colnames(probs)) && ncol(probs) == length(levels_order)) {
      colnames(probs) <- levels_order
    }
    preds <- apply(probs, 1, function(r) {
      lvl <- colnames(probs)[which.max(r)]
      if (is.null(lvl)) return(which.max(r) - 1)
      lvl
    })
    return(list(labels = factor(preds, levels = levels_order), probs = probs))
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

    start_time <- Sys.time()
    pred_data <- predict_labels(model, X_test, levels_order)
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
