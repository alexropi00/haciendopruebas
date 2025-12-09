# ============================================================================
# 04_repair_ultralight.R - REPARACIÓN DE EMERGENCIA (LOW RAM)
# ============================================================================

library(nnet)
library(e1071)
library(randomForest)

# Función de limpieza profunda
strip_model <- function(model) {
  if (inherits(model, "svm")) {
    model$fitted <- NULL; model$residuals <- NULL
  }
  if (inherits(model, "nnet")) {
    model$fitted.values <- NULL; model$residuals <- NULL
  }
  if (inherits(model, "randomForest")) {
    model$y <- NULL; model$predicted <- NULL; model$votes <- NULL
  }
  return(model)
}

get_probs_named <- function(model, X, all_levels) {
  raw_pred <- tryCatch({
    if (inherits(model, "nnet")) {
      predict(model, X, type = "raw")
    } else if (inherits(model, "svm")) {
      attr(predict(model, X, probability = TRUE), "probabilities")
    } else if (inherits(model, "randomForest")) {
      predict(model, X, type = "prob")
    } else { NULL }
  }, error = function(e) NULL)
  
  if (is.null(raw_pred)) return(NULL)
  
  probs <- raw_pred
  if (is.vector(probs)) probs <- matrix(probs, ncol = 1)
  
  n_rows <- nrow(X)
  n_cols <- length(all_levels)
  full_probs <- matrix(0, nrow = n_rows, ncol = n_cols)
  colnames(full_probs) <- all_levels
  
  if (!is.null(colnames(probs))) {
    available_cols <- intersect(colnames(probs), all_levels)
    if (length(available_cols) > 0) full_probs[, available_cols] <- probs[, available_cols]
  } else if (ncol(probs) == n_cols) {
    full_probs[] <- probs
  }
  return(full_probs)
}

repair_ensembles <- function(dataset_path, model_path, n_samples = 1000) {
  cat(sprintf("\n>>> Reparando (ULTRALIGHT) para: %s <<<\n", model_path))
  
  # --- 1. CARGAR DATOS Y EXTRAER MUESTRA ---
  cat("1. Cargando datos...\n")
  load(dataset_path)
  
  if (exists("pca_result")) {
    full_train <- pca_result$train
    rm(pca_result)
  } else {
    full_train <- crop_result$train
    rm(crop_result)
  }
  
  if (!exists("train_labels")) stop("Falta train_labels")
  
  set.seed(123)
  idx <- sample(1:nrow(full_train), min(n_samples, nrow(full_train)))
  
  X_ens <- full_train[idx, ]
  y_ens <- as.factor(train_labels[idx])
  all_levels <- levels(y_ens)
  
  # LIBERAR MEMORIA AGRESIVAMENTE
  cat("   (Liberando dataset original...)\n")
  rm(full_train, train_labels)
  gc()
  
  # --- 2. CARGAR MODELOS ---
  cat("2. Cargando modelos...\n")
  load(model_path)
  
  if (exists("models_pca")) {
    models <- models_pca
    obj_name <- "models_pca"
    rm(models_pca)
  } else {
    models <- models_crop
    obj_name <- "models_crop"
    rm(models_crop)
  }
  
  base_names <- names(models)[!grepl("^ensemble_", names(models))]
  base_names <- base_names[sapply(base_names, function(n) !is.null(models[[n]]$model))]
  
  cat(sprintf("   Usando %d modelos base.\n", length(base_names)))
  
  # --- 3. GENERAR META-FEATURES "IN-PLACE" ---
  cat("3. Generando meta-features (Modo seguro)...\n")
  
  # Pre-alocar la matriz gigante de una vez (es más eficiente que crecer una lista)
  # n_samples filas x (n_modelos * 10 clases) columnas
  total_cols <- length(base_names) * length(all_levels)
  X_meta <- matrix(0, nrow = nrow(X_ens), ncol = total_cols)
  
  for (i in seq_along(base_names)) {
    name <- base_names[i]
    cat(".")
    
    # Obtener probabilidades
    p <- get_probs_named(models[[name]]$model, X_ens, all_levels)
    
    if (!is.null(p)) {
      # Calcular índices de columnas en la matriz gigante
      col_start <- (i - 1) * length(all_levels) + 1
      col_end   <- i * length(all_levels)
      
      # Insertar directamente
      X_meta[, col_start:col_end] <- p
    }
    
    # Forzar limpieza tras cada modelo
    rm(p)
    gc()
  }
  cat("\n")
  
  # --- 4. ENTRENAR ENSEMBLES ---
  cat("4. Entrenando ensembles...\n")
  ens_mlp <- nnet(X_meta, class.ind(y_ens), size = 20, maxit = 100, trace = F, softmax = T, MaxNWts = 50000)
  ens_rf <- randomForest(X_meta, y_ens, ntree = 30) # Reducido a 30 árboles para ahorrar RAM
  ens_svm <- svm(X_meta, y_ens, kernel = "radial", probability = TRUE)
  
  # Limpiar modelos (stripping)
  ens_mlp <- strip_model(ens_mlp)
  ens_rf <- strip_model(ens_rf)
  ens_svm <- strip_model(ens_svm)
  
  # --- 5. GUARDAR ---
  cat("5. Guardando...\n")
  
  # Limpiar modelos base existentes también
  for (nm in base_names) {
    models[[nm]]$model <- strip_model(models[[nm]]$model)
  }
  
  models[["ens_mlp"]] <- list(model = ens_mlp, type = "ensemble_mlp", base_model_names = base_names)
  models[["ens_rf"]] <- list(model = ens_rf, type = "ensemble_rf", base_model_names = base_names)
  models[["ens_svm"]] <- list(model = ens_svm, type = "ensemble_svm", base_model_names = base_names)
  
  assign(obj_name, models)
  save(list = c(obj_name), file = model_path, compress = "xz")
  
  cat("✓ Guardado OK.\n")
  rm(models, X_meta, X_ens, y_ens)
  gc()
}

main <- function() {
  # PCA: 5000 muestras
  repair_ensembles("processed_data/mnist_pca.RData", "trained_models/models_pca.RData", 1500)
  
  # CROP: SOLO 1000 MUESTRAS (Suficiente para que el ensemble aprenda a ponderar)
  repair_ensembles("processed_data/mnist_crop_pooling.RData", "trained_models/models_crop_pooling.RData", 1500)
}

main()
