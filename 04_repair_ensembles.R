# ============================================================================
# 04_repair_ensembles.R - RECONSTRUCCIÓN DE ENSEMBLES (PARALELIZADO)
# ============================================================================

library(nnet)
library(e1071)
library(randomForest)
library(parallel)

# ---------------------------------------------------------------------------
# Función de limpieza profunda
# ---------------------------------------------------------------------------

strip_model <- function(model) {
  if (inherits(model, "svm")) {
    model$fitted <- NULL
    model$residuals <- NULL
  }
  if (inherits(model, "nnet")) {
    model$fitted.values <- NULL
    model$residuals <- NULL
  }
  if (inherits(model, "randomForest")) {
    model$y <- NULL
    model$predicted <- NULL
    model$votes <- NULL
  }
  return(model)
}

# ---------------------------------------------------------------------------
# Obtiene probabilidades y las alinea con todas las clases esperadas
# ---------------------------------------------------------------------------

get_probs_named <- function(model, X, all_levels) {
  # Asegurar formato data.frame numérico
  X_df <- as.data.frame(as.matrix(X))
  n_rows <- nrow(X_df)
  n_classes <- length(all_levels)
  
  # 1) Sacar probabilidades con predict()
  safe_pred <- tryCatch({
    if (inherits(model, "nnet")) {
      p <- predict(model, X_df, type = "raw")
      if (is.vector(p)) {
        # Vector plano -> levantar a matriz n_rows x ?
        p <- matrix(p, nrow = n_rows, byrow = TRUE)
      }
      p
    } else if (inherits(model, "svm")) {
      attr(predict(model, X_df, probability = TRUE), "probabilities")
    } else if (inherits(model, "randomForest")) {
      predict(model, X_df, type = "prob")
    } else {
      # Tipo de modelo no soportado
      return(NULL)
    }
  }, error = function(e) {
    cat(sprintf(
      "\n   [DEBUG] get_probs_named: predict() falló para modelo de clase [%s]: %s\n",
      paste(class(model), collapse = ","),
      e$message
    ))
    NULL
  })
  
  if (is.null(safe_pred)) return(NULL)
  
  # 2) Convertir a matriz
  probs <- tryCatch(as.matrix(safe_pred), error = function(e) {
    cat(sprintf("\n   [DEBUG] get_probs_named: as.matrix() falló: %s\n", e$message))
    NULL
  })
  if (is.null(probs)) return(NULL)
  
  # 3) Normalizar dimensiones
  if (is.null(dim(probs))) {
    # Vector sin dim: intentar levantar a n_rows x 1
    if (length(probs) == n_rows) {
      probs <- matrix(probs, nrow = n_rows, ncol = 1)
    } else {
      cat(sprintf(
        "\n   [DEBUG] get_probs_named: vector de longitud %d incompatible con n_rows = %d.\n",
        length(probs), n_rows
      ))
      return(NULL)
    }
  }
  
  if (nrow(probs) != n_rows) {
    cat(sprintf(
      "\n   [DEBUG] get_probs_named: nrow(probs) = %d != n_rows = %d.\n",
      nrow(probs), n_rows
    ))
    return(NULL)
  }
  
  # Normalizar NAs
  probs[is.na(probs)] <- 0
  
  # 4) Alinear con el conjunto completo de clases
  full_probs <- matrix(0, nrow = n_rows, ncol = n_classes)
  colnames(full_probs) <- all_levels
  
  if (!is.null(colnames(probs))) {
    available <- intersect(colnames(probs), all_levels)
    if (length(available) > 0) {
      full_probs[, available] <- probs[, available, drop = FALSE]
    } else {
      cat(sprintf(
        "\n   [DEBUG] get_probs_named: colnames(probs) = c(%s) no coincide con ninguna clase esperada.\n",
        paste(colnames(probs), collapse = ", ")
      ))
      return(NULL)
    }
  } else if (ncol(probs) == n_classes) {
    # Sin nombres pero mismo nº de columnas: asumimos mismo orden
    full_probs[] <- probs
  } else {
    cat(sprintf(
      "\n   [DEBUG] get_probs_named: ncol(probs) = %d != n_classes = %d y sin nombres de columnas.\n",
      ncol(probs), n_classes
    ))
    return(NULL)
  }
  
  full_probs
}

# ---------------------------------------------------------------------------
# Barra de progreso para pasos globales
# ---------------------------------------------------------------------------

make_step_tracker <- function(total_steps) {
  start_time <- Sys.time()
  pb <- txtProgressBar(min = 0, max = total_steps, style = 3)
  current_step <- 0
  
  list(
    advance = function(label) {
      current_step <<- current_step + 1
      setTxtProgressBar(pb, current_step)
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      eta <- max(0, (elapsed / current_step) * (total_steps - current_step))
      cat(sprintf(
        "\n   Paso %d/%d: %s | Tiempo transcurrido: %.1fs | ETA restante: %.1fs\n",
        current_step, total_steps, label, elapsed, eta
      ))
    },
    close = function() close(pb)
  )
}

# ---------------------------------------------------------------------------
# Calcula la matriz de meta-features en paralelo (si hay núcleos disponibles)
# ---------------------------------------------------------------------------

build_meta_matrix <- function(base_names, models, X_ens, all_levels, cores) {
  total_cols <- length(base_names) * length(all_levels)
  X_meta <- matrix(0, nrow = nrow(X_ens), ncol = total_cols)
  colnames(X_meta) <- as.vector(t(outer(base_names, all_levels, paste, sep = "__")))
  
  step_start <- Sys.time()
  pb <- txtProgressBar(min = 0, max = length(base_names), style = 3)
  update_bar <- function(i) {
    setTxtProgressBar(pb, i)
    elapsed <- as.numeric(difftime(Sys.time(), step_start, units = "secs"))
    if (i > 0) {
      eta <- max(0, (elapsed / i) * (length(base_names) - i))
      cat(sprintf(
        "\r   Modelos procesados: %d/%d | Tiempo: %.1fs | ETA: %.1fs",
        i, length(base_names), elapsed, eta
      ))
    }
  }
  
  if (length(base_names) == 0) {
    stop("No hay modelos base válidos para construir el ensemble.")
  }
  
  workload <- nrow(X_ens) * length(base_names)
  use_parallel <- cores > 1 && length(base_names) > 1 && workload >= 2500
  
  # Exportamos solo los modelos base para minimizar el tráfico al cluster
  base_models <- lapply(base_names, function(nm) models[[nm]]$model)
  
  run_seq <- function() {
    lapply(seq_along(base_names), function(i) {
      name <- base_names[[i]]
      p <- tryCatch(
        get_probs_named(models[[name]]$model, X_ens, all_levels),
        error = function(e) {
          cat(sprintf("\n   [WARN] Modelo %s falló: %s\n", name, e$message))
          NULL
        }
      )
      update_bar(i)
      list(idx = i, probs = p)
    })
  }
  
  if (use_parallel) {
    cl <- makeCluster(cores)
    on.exit(stopCluster(cl), add = TRUE)
    clusterExport(
      cl,
      varlist = c("base_models", "X_ens", "all_levels", "get_probs_named", "base_names"),
      envir = environment()
    )
    clusterEvalQ(cl, {library(nnet); library(e1071); library(randomForest)})
    
    results <- tryCatch({
      parLapply(cl, seq_along(base_names), function(i) {
        name <- base_names[[i]]
        p <- tryCatch(
          get_probs_named(base_models[[i]], X_ens, all_levels),
          error = function(e) NULL
        )
        if (is.null(p)) {
          list(idx = i, probs = NULL, warn = sprintf("Modelo %s sin probabilidades válidas", name))
        } else {
          list(idx = i, probs = p, warn = NULL)
        }
      })
    }, error = function(e) {
      cat(sprintf(
        "\n   [INFO] Paralelización falló (%s). Reintentando en modo secuencial...\n",
        e$message
      ))
      run_seq()
    })
    update_bar(length(base_names))
    cat("\n")
  } else {
    if (cores > 1 && workload < 2500) {
      cat(sprintf(
        "\n   [INFO] Carga pequeña (%d unidades); se usa ejecución secuencial para evitar sobrecoste de cluster.\n",
        workload
      ))
    }
    results <- run_seq()
    cat("\n")
  }
  
  for (res in results) {
    if (!is.null(res$warn)) cat(sprintf("\n   [WARN] %s\n", res$warn))
    if (is.null(res$probs)) next
    if (nrow(res$probs) != nrow(X_meta)) {
      cat(sprintf(
        "\n   [WARN] Dimensiones incompatibles para modelo %s; se omite.\n",
        base_names[[res$idx]]
      ))
      next
    }
    if (ncol(res$probs) != length(all_levels)) {
      cat(sprintf(
        "\n   [WARN] Probabilidades incompletas para modelo %s; se omite.\n",
        base_names[[res$idx]]
      ))
      next
    }
    
    col_start <- (res$idx - 1) * length(all_levels) + 1
    col_end <- col_start + length(all_levels) - 1
    X_meta[, col_start:col_end] <- res$probs
  }
  
  close(pb)
  X_meta
}

# ---------------------------------------------------------------------------
# Muestreo estratificado para el ensemble
# ---------------------------------------------------------------------------

sample_ensemble_data <- function(full_train, train_labels, n_samples) {
  all_levels <- levels(factor(train_labels))
  n_take <- min(max(n_samples, length(all_levels)), nrow(full_train))
  
  idx_by_level <- lapply(all_levels, function(lv) which(train_labels == lv))
  names(idx_by_level) <- all_levels
  
  missing_levels <- names(idx_by_level)[sapply(idx_by_level, length) == 0]
  if (length(missing_levels) > 0) {
    warning(sprintf(
      "Clases ausentes en el dataset original: %s",
      paste(missing_levels, collapse = ", ")
    ))
  }
  
  present_levels <- setdiff(all_levels, missing_levels)
  per_class <- lapply(idx_by_level[present_levels], function(idx_lv) sample(idx_lv, 1))
  remaining <- n_take - length(per_class)
  
  pool <- unlist(idx_by_level[present_levels], use.names = FALSE)
  extra <- if (remaining > 0) {
    sample(pool, remaining, replace = length(pool) < remaining)
  } else {
    integer(0)
  }
  
  idx <- sample(c(unlist(per_class), extra))
  y_ens <- factor(train_labels[idx], levels = present_levels)
  
  list(idx = idx, y = y_ens, levels = levels(y_ens))
}

# ---------------------------------------------------------------------------
# Reparación de ensembles
# ---------------------------------------------------------------------------

repair_ensembles <- function(dataset_path, model_path,
                             n_samples = 1000,
                             cores = max(1, detectCores() - 1)) {
  cat(sprintf("\n>>> Reparando ensembles para: %s <<<\n", model_path))
  tracker <- make_step_tracker(total_steps = 5)
  on.exit(tracker$close(), add = TRUE)
  
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
  sampled <- sample_ensemble_data(full_train, train_labels, n_samples)
  
  X_ens <- as.matrix(full_train[sampled$idx, , drop = FALSE])
  y_ens <- sampled$y
  class_levels <- sampled$levels
  
  cat(sprintf(
    "   Muestras tomadas: %d | Clases: %d\n",
    nrow(X_ens), length(class_levels)
  ))
  tracker$advance("Datos cargados y muestreados")
  
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
  
  # Excluir tanto ensembles nuevos (ensemble_*) como antiguos (ens_*)
  base_names <- names(models)
  base_names <- base_names[!grepl("^(ensemble_|ens_)", base_names)]
  base_names <- base_names[sapply(base_names, function(n) !is.null(models[[n]]$model))]
  
  cat(sprintf("   Usando %d modelos base.\n", length(base_names)))
  tracker$advance("Modelos base cargados")
  
  # --- 3. GENERAR META-FEATURES ---
  cat("3. Generando meta-features...")
  X_meta <- build_meta_matrix(base_names, models, X_ens, class_levels, cores)
  cat(" listo.\n")
  tracker$advance("Meta-features generados")
  
  # --- 4. ENTRENAR ENSEMBLES ---
  cat("4. Entrenando ensembles (MLP, RF, SVM)...\n")
  y_ens <- droplevels(y_ens)
  
  ens_mlp <- nnet(
    x = X_meta, y = class.ind(y_ens),
    size = 40, maxit = 200, trace = FALSE,
    softmax = TRUE, MaxNWts = 75000
  )
  ens_rf <- randomForest(
    x = X_meta, y = y_ens,
    ntree = 80, mtry = max(1, floor(sqrt(ncol(X_meta)))),
    importance = FALSE
  )
  ens_svm <- svm(
    x = X_meta, y = y_ens,
    kernel = "radial", cost = 5,
    gamma = 1 / ncol(X_meta), probability = TRUE
  )
  tracker$advance("Ensembles entrenados")
  
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
  
  models[["ensemble_mlp"]] <- list(
    model = ens_mlp,
    type = "ensemble_mlp",
    base_model_names = base_names
  )
  models[["ensemble_rf"]] <- list(
    model = ens_rf,
    type = "ensemble_rf",
    base_model_names = base_names
  )
  models[["ensemble_svm"]] <- list(
    model = ens_svm,
    type = "ensemble_svm",
    base_model_names = base_names
  )
  
  assign(obj_name, models)
  save(list = c(obj_name), file = model_path, compress = "xz")
  
  cat("✓ Guardado OK.\n")
  tracker$advance("Modelos guardados")
  rm(models, X_meta, X_ens, y_ens)
  gc()
}

# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

main <- function() {
  # Primero reparamos PCA con un solo núcleo (sin paralelización)
  #repair_ensembles(
  #  "processed_data/mnist_pca.RData",
  #  "trained_models/models_pca.RData",
  #  n_samples = 1500,
  #  cores = 1
  # )
  
  # Cuando PCA vaya fino puedes descomentar esto para el CROP:
  repair_ensembles(
    "processed_data/mnist_crop_pooling.RData",
    "trained_models/models_crop_pooling.RData",
    n_samples = 1500,
    cores = 1
  )
}

# Ejecuta main() cuando quieras lanzar la reparación:
main()
