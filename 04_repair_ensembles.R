# ============================================================================
# 04_repair_ensembles.R - RECONSTRUCCIÓN DE ENSEMBLES (PARALELIZADO)
# ============================================================================

library(nnet)
library(e1071)
library(randomForest)
library(parallel)

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

# Obtiene probabilidades y las alinea con todas las clases esperadas
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
    if (length(available_cols) > 0) full_probs[, available_cols, drop = FALSE] <- probs[, available_cols, drop = FALSE]
  } else if (ncol(probs) == n_cols) {
    full_probs[] <- probs
  }
  return(full_probs)
}

# Calcula la matriz de meta-features en paralelo (si hay núcleos disponibles)
build_meta_matrix <- function(base_names, models, X_ens, all_levels, cores) {
  total_cols <- length(base_names) * length(all_levels)
  X_meta <- matrix(0, nrow = nrow(X_ens), ncol = total_cols)

  if (length(base_names) == 0) {
    stop("No hay modelos base válidos para construir el ensemble.")
  }

  use_parallel <- cores > 1 && length(base_names) > 1

  if (use_parallel) {
    cl <- makeCluster(cores)
    on.exit(stopCluster(cl), add = TRUE)
    clusterExport(cl, varlist = c("models", "X_ens", "all_levels", "get_probs_named"), envir = environment())
    clusterEvalQ(cl, {library(nnet); library(e1071); library(randomForest)})

    results <- parLapply(cl, base_names, function(name) {
      p <- get_probs_named(models[[name]]$model, X_ens, all_levels)
      list(name = name, probs = p)
    })
  } else {
    results <- lapply(base_names, function(name) {
      cat(".")
      p <- get_probs_named(models[[name]]$model, X_ens, all_levels)
      list(name = name, probs = p)
    })
  }

  for (i in seq_along(results)) {
    name <- results[[i]]$name
    p <- results[[i]]$probs
    if (is.null(p)) next

    col_start <- (which(base_names == name) - 1) * length(all_levels) + 1
    col_end <- col_start + length(all_levels) - 1
    X_meta[, col_start:col_end] <- p
  }

  X_meta
}

repair_ensembles <- function(dataset_path, model_path, n_samples = 1000, cores = max(1, detectCores() - 1)) {
  cat(sprintf("\n>>> Reparando ensembles para: %s <<<\n", model_path))

  # --- 1. CARGAR DATOS Y EXTRAER MUESTRA ---
  cat("1. Cargando datos...\n")
  load(dataset_path)

  if (exists("pca_result")) {
    full_train <- pca_result$train
    rm(pca_result)
  } else if (exists("crop_result")) {
    full_train <- crop_result$train
    rm(crop_result)
  } else {
    stop("El RData no contiene 'pca_result' ni 'crop_result'.")
  }

  if (!exists("train_labels")) stop("Falta 'train_labels' en el dataset cargado.")

  set.seed(123)
  n_take <- min(n_samples, nrow(full_train))
  idx <- sample(seq_len(nrow(full_train)), n_take)

  X_ens <- full_train[idx, ]
  y_ens <- factor(train_labels[idx])
  all_levels <- levels(y_ens)

  cat(sprintf("   Muestras tomadas: %d | Clases: %d\n", nrow(X_ens), length(all_levels)))

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
  } else if (exists("models_crop")) {
    models <- models_crop
    obj_name <- "models_crop"
    rm(models_crop)
  } else {
    stop("El archivo de modelos no contiene 'models_pca' ni 'models_crop'.")
  }

  base_names <- names(models)[!grepl("^ensemble_", names(models))]
  base_names <- base_names[sapply(base_names, function(n) !is.null(models[[n]]$model))]

  cat(sprintf("   Usando %d modelos base.\n", length(base_names)))

  # --- 3. GENERAR META-FEATURES ---
  cat("3. Generando meta-features...")
  X_meta <- build_meta_matrix(base_names, models, X_ens, all_levels, cores)
  cat(" listo.\n")

  # --- 4. ENTRENAR ENSEMBLES ---
  cat("4. Entrenando ensembles (MLP, RF, SVM)...\n")
  ens_mlp <- nnet(x = X_meta, y = class.ind(y_ens), size = 40, maxit = 200, trace = FALSE, softmax = TRUE, MaxNWts = 75000)
  ens_rf <- randomForest(x = X_meta, y = y_ens, ntree = 80, mtry = max(1, floor(sqrt(ncol(X_meta)))), importance = FALSE)
  ens_svm <- svm(x = X_meta, y = y_ens, kernel = "radial", cost = 5, gamma = 1 / ncol(X_meta), probability = TRUE)

  # Limpiar modelos (stripping)
  ens_mlp <- strip_model(ens_mlp)
  ens_rf <- strip_model(ens_rf)
  ens_svm <- strip_model(ens_svm)

  # --- 5. GUARDAR ---
  cat("5. Guardando...\n")
  dir.create(dirname(model_path), showWarnings = FALSE, recursive = TRUE)

  # Limpiar modelos base existentes también
  for (nm in base_names) {
    models[[nm]]$model <- strip_model(models[[nm]]$model)
  }

  models[["ensemble_mlp" ]] <- list(model = ens_mlp, type = "ensemble_mlp", base_model_names = base_names)
  models[["ensemble_rf"  ]] <- list(model = ens_rf, type = "ensemble_rf", base_model_names = base_names)
  models[["ensemble_svm" ]] <- list(model = ens_svm, type = "ensemble_svm", base_model_names = base_names)

  assign(obj_name, models)
  save(list = c(obj_name), file = model_path, compress = "xz")

  cat("✓ Guardado OK.\n")
  rm(models, X_meta, X_ens, y_ens)
  gc()
}

main <- function() {
  # PCA
  repair_ensembles("processed_data/mnist_pca.RData", "trained_models/models_pca.RData", n_samples = 1500)

  # CROP
  repair_ensembles("processed_data/mnist_crop_pooling.RData", "trained_models/models_crop_pooling.RData", n_samples = 1500)
}

main()
