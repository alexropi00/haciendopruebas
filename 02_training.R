# ============================================================================
# 02_training.R - PARALELISMO ROBUSTO (FIX SERIALIZACION & ENVIRONMENTS)
# ============================================================================

library(nnet)
library(e1071)
library(randomForest)
library(parallel)

# ============================================================================
# PROGRESO EN PARALELISMO
# ============================================================================
execute_jobs_with_progress <- function(cl, jobs, worker_fun) {
  total_jobs <- length(jobs)
  if (total_jobs == 0) return(list())

  # Barra de progreso básica
  pb <- txtProgressBar(min = 0, max = total_jobs, style = 3)
  start_time <- Sys.time()

  results <- vector("list", total_jobs)
  next_job <- 1

  # Enviar una tarea por worker al inicio
  for (node in seq_along(cl)) {
    if (next_job <= total_jobs) {
      parallel:::sendCall(cl[[node]], worker_fun, list(jobs[[next_job]]), tag = next_job)
      next_job <- next_job + 1
    }
  }

  completed <- 0

  while (completed < total_jobs) {
    res <- parallel:::recvOneResult(cl)
    completed <- completed + 1
    results[[res$tag]] <- res$value

    # Actualizar progreso y ETA
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    avg_time <- elapsed / completed
    remaining <- total_jobs - completed
    eta_hours <- (avg_time * remaining) / 3600

    setTxtProgressBar(pb, completed)
    cat(sprintf("  [%d/%d] ETA: %.2f horas (%.1f min transcurridos)\n",
                completed, total_jobs, eta_hours, elapsed / 60))

    # Enviar siguiente tarea al mismo nodo si quedan pendientes
    if (next_job <= total_jobs) {
      parallel:::sendCall(cl[[res$node]], worker_fun, list(jobs[[next_job]]), tag = next_job)
      next_job <- next_job + 1
    }
  }

  close(pb)
  results
}

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
MAX_CORES <- 10
SUBSET_PERCENTAGE <- 0.40
RF_NTREES <- 50
MLP_MAX_ITER <- 300
ENSEMBLE_MAX_ITER <- 200

# Directorio temporal para intercambio de archivos
TEMP_DIR <- "temp_models_windows"
if (dir.exists(TEMP_DIR)) unlink(TEMP_DIR, recursive = TRUE)
dir.create(TEMP_DIR)

cat(sprintf("Modo: Cluster PSOCK (Windows Compatible)\nNúcleos: %d\n", MAX_CORES))

# ============================================================================
# FUNCIÓN WORKER (Definida limpia)
# ============================================================================
worker_wrapper <- function(job) {
  # Pequeño retraso aleatorio para evitar que los 10 núcleos lean el disco 
  # en el mismo milisegundo exacto (evita I/O timeout)
  Sys.sleep(runif(1, 0.1, 2.0))
  
  # Limpieza de memoria
  gc()
  
  # Cargar librerías explícitamente dentro del worker
  library(nnet)
  library(e1071)
  library(randomForest)
  
  # Extraer datos del job
  config <- job$config
  seed <- job$seed
  dataset_path <- job$dataset_path
  dataset_type <- job$dataset_type
  
  # --- CACHÉ DE DATOS EN EL WORKER ---
  # Usamos el entorno global del worker para persistir datos entre jobs
  # Esto evita recargar el archivo RData en cada iteración
  if (!exists(".WORKER_DATA_CACHE", envir = .GlobalEnv)) {
    # Cargar archivo
    load(dataset_path) # Crea 'pca_result' o 'crop_result' y 'train_labels'
    
    if (dataset_type == "pca") {
      .GlobalEnv$.WORKER_DATA_CACHE <- list(X = pca_result$train, y = train_labels)
    } else {
      .GlobalEnv$.WORKER_DATA_CACHE <- list(X = crop_result$train, y = train_labels)
    }
    # Limpiar variables temporales de la carga
    rm(list = c("pca_result", "crop_result", "train_labels", "dataset_path"))
    gc()
  }
  
  # --- PREPARAR SUBSET ---
  set.seed(seed)
  cache <- .GlobalEnv$.WORKER_DATA_CACHE
  
  # Muestreo estratificado (rápido)
  # Usamos índices para no duplicar memoria innecesariamente
  subset_percentage <- job$subset_percentage
  if (!(subset_percentage > 0 && subset_percentage <= 1)) {
    stop(sprintf("subset_percentage debe estar en el rango (0, 1]; valor recibido: %.3f", subset_percentage))
  }

  train_idx <- c()
  for (cls in unique(cache$y)) {
    c_idx <- which(cache$y == cls)
    n <- max(2, floor(length(c_idx) * subset_percentage))
    train_idx <- c(train_idx, sample(c_idx, n))
  }

  X_sub <- cache$X[train_idx, ]
  y_sub <- as.factor(cache$y[train_idx])
  y_sub_matrix <- class.ind(y_sub) # Matrix mantiene alineación fila-fila con X_sub
  
  # --- ENTRENAMIENTO ---
  start <- Sys.time()
  success <- FALSE
  error_msg <- ""
  model <- NULL
  
  tryCatch({
    if (config$type == "mlp") {
      if (!is.numeric(config$hidden) || length(config$hidden) != 1) {
        stop("config$hidden debe ser un valor numérico escalar para 'nnet'.")
      }
      model <- nnet(x = X_sub, y = y_sub_matrix, size = config$hidden, maxit = config$max_iter,
                    decay = config$decay, MaxNWts = 50000, trace = FALSE, softmax = TRUE)
    } else if (config$type == "svm") {
      if (is.null(config$gamma)) {
        model <- svm(x = X_sub, y = y_sub, kernel = config$kernel, cost = config$cost, scale = TRUE, probability = TRUE)
      } else {
        model <- svm(x = X_sub, y = y_sub, kernel = config$kernel, cost = config$cost, gamma = config$gamma, scale = TRUE, probability = TRUE)
      }
    }
    success <- TRUE
  }, error = function(e) error_msg <<- e$message)
  
  elapsed <- as.numeric(difftime(Sys.time(), start, units = "secs"))
  
  # --- GUARDADO EN DISCO ---
  if (success) {
    out_file <- paste0(job$temp_dir, "/", dataset_type, "_", config$name, ".RData")
    save(model, file = out_file)
    
    return(list(success = TRUE, path = out_file, name = config$name, 
                type = config$type, time = elapsed, config = config))
  } else {
    return(list(success = FALSE, name = config$name, error = error_msg))
  }
}

# !!! IMPORTANTE: LIMPIAR EL ENTORNO DE LA FUNCIÓN !!!
# Esto evita que 'worker_wrapper' arrastre todo tu GlobalEnv (datasets grandes) al cluster
environment(worker_wrapper) <- new.env() 

# ============================================================================
# GESTOR DE PARALELISMO
# ============================================================================
run_parallel_training <- function(dataset_name, dataset_path, subset_percentage = SUBSET_PERCENTAGE) {

  if (!(subset_percentage > 0 && subset_percentage <= 1)) {
    stop(sprintf("SUBSET_PERCENTAGE debe estar en el rango (0, 1]; valor recibido: %.3f", subset_percentage))
  }
  
  cat(sprintf("\n>>> PROCESANDO: %s <<<\n", toupper(dataset_name)))
  
  # Configs
  configs <- list(
    list(type = "mlp", name = "mlp_tiny", hidden = 20, decay = 0.001, max_iter = MLP_MAX_ITER),
    list(type = "mlp", name = "mlp_small", hidden = 40, decay = 0.001, max_iter = MLP_MAX_ITER),
    list(type = "mlp", name = "mlp_medium", hidden = 60, decay = 0.001, max_iter = MLP_MAX_ITER),
    list(type = "mlp", name = "mlp_large", hidden = 80, decay = 0.001, max_iter = MLP_MAX_ITER),
    list(type = "mlp", name = "mlp_deep", hidden = 100, decay = 0.001, max_iter = MLP_MAX_ITER),
    list(type = "svm", name = "svm_rad_01", kernel = "radial", cost = 0.1, gamma = NULL),
    list(type = "svm", name = "svm_rad_1", kernel = "radial", cost = 1, gamma = NULL),
    list(type = "svm", name = "svm_rad_5", kernel = "radial", cost = 5, gamma = NULL),
    list(type = "svm", name = "svm_poly", kernel = "polynomial", cost = 1, gamma = NULL),
    list(type = "svm", name = "svm_lin", kernel = "linear", cost = 1, gamma = NULL)
  )
  
  # Crear Jobs (Ligeros, solo texto)
  jobs <- lapply(seq_along(configs), function(i) {
    list(
      config = configs[[i]],
      seed = i * 1000,
      dataset_path = dataset_path,
      dataset_type = dataset_name,
      temp_dir = TEMP_DIR,
      subset_percentage = subset_percentage
    )
  })
  
  # Iniciar Cluster
  detected_cores <- detectCores()
  n_cores_use <- min(MAX_CORES, max(1, detected_cores - 1))
  cat(sprintf("Detectados %d núcleos, usando %d para clusterApplyLB\n", detected_cores, n_cores_use))
  cl <- makeCluster(n_cores_use, outfile = "")
  on.exit(stopCluster(cl))
  
  cat("Enviando jobs a workers (Environment Cleaned)...\n")

  # Ejecutar con Balanceo de Carga mostrando progreso y ETA
  results_meta <- execute_jobs_with_progress(cl, jobs, worker_wrapper)
  
  # Recuperar resultados (Solo rutas de archivos)
  cat("Recuperando modelos desde disco...\n")
  final_models <- list()
  
  for (res in results_meta) {
    if (res$success) {
      load(res$path) # Carga 'model'
      final_models[[res$name]] <- list(
        model = model,
        type = res$type,
        config = res$config,
        success = TRUE,
        time = res$time
      )
      cat(sprintf("✓ %s (%.1fs)\n", res$name, res$time))
      unlink(res$path) # Borrar archivo temporal
    } else {
      cat(sprintf("✗ %s falló: %s\n", res$name, res$error))
    }
  }
  
  return(final_models)
}

# ============================================================================
# HELPER PROBABILIDADES
# ============================================================================
get_probs <- function(model, X) {
  tryCatch({
    if (inherits(model, "nnet")) {
      p <- predict(model, X, type = "raw")
      if(is.vector(p)) p <- matrix(p, nrow=nrow(X), byrow=T)
    } else {
      p <- attr(predict(model, X, probability=T), "probabilities")
    }
    p[is.na(p)] <- 0
    return(p)
  }, error = function(e) NULL)
}

# ============================================================================
# MAIN
# ============================================================================
main <- function(subset_percentage = SUBSET_PERCENTAGE) {
  if (!dir.exists("trained_models")) dir.create("trained_models")

  ens_list <- list(
    list(n="ens_mlp", t="mlp"),
    list(n="ens_rf", t="rf"),
    list(n="ens_svm", t="svm")
  )
  
  # --- PCA ---
  # Importante: NO cargar los datos en el entorno global antes de llamar al cluster
  # Dejar que los workers lo hagan.
  
  models_pca <- run_parallel_training("pca", "processed_data/mnist_pca.RData", subset_percentage)
  
  # Ensembles PCA
  cat("Entrenando Ensembles PCA...\n")
  load("processed_data/mnist_pca.RData") # Cargar SOLO AHORA para ensembles
  
  successful <- models_pca[sapply(models_pca, function(x) x$success)]
  if(length(successful) >= 3) {
    selected <- successful[1:min(5, length(successful))]
    set.seed(123)
    idx <- sample(1:nrow(pca_result$train), 5000)
    X_ens <- pca_result$train[idx,]
    y_ens <- train_labels[idx]
    
    meta <- matrix(0, nrow=length(y_ens), ncol=length(selected)*10)
    for(i in seq_along(selected)) {
      p <- get_probs(selected[[i]]$model, X_ens)
      if(!is.null(p)) meta[, ((i-1)*10+1):((i-1)*10+ncol(p))] <- p
    }
    
    if(length(ens_list) > 0) {
      for(e in ens_list) {
        if(e$t=="mlp") m <- nnet(meta, y_ens, size=30, maxit=200, trace=F, MaxNWts = 50000)
        if(e$t=="rf") m <- randomForest(meta, y_ens, ntree=50)
        if(e$t=="svm") m <- svm(meta, y_ens, probability=T)
        models_pca[[e$n]] <- list(model=m, success=TRUE, type=paste0("ensemble_", e$t))
        cat(sprintf("✓ %s\n", e$n))
      }
    }
  }
  save(models_pca, file="trained_models/models_pca.RData")
  rm(models_pca, pca_result); gc()
  
  # --- CROP ---
  models_crop <- run_parallel_training("crop", "processed_data/mnist_crop_pooling.RData", subset_percentage)
  
  # Ensembles Crop
  cat("Entrenando Ensembles Crop...\n")
  load("processed_data/mnist_crop_pooling.RData") # Cargar SOLO AHORA
  
  successful <- models_crop[sapply(models_crop, function(x) x$success)]
  if(length(successful) >= 3) {
    selected <- successful[1:min(5, length(successful))]
    set.seed(123)
    idx <- sample(1:nrow(crop_result$train), 5000)
    X_ens <- crop_result$train[idx,]
    y_ens <- train_labels[idx]
    
    meta <- matrix(0, nrow=length(y_ens), ncol=length(selected)*10)
    for(i in seq_along(selected)) {
      p <- get_probs(selected[[i]]$model, X_ens)
      if(!is.null(p)) meta[, ((i-1)*10+1):((i-1)*10+ncol(p))] <- p
    }
    
    if(length(ens_list) > 0) {
      for(e in ens_list) {
        if(e$t=="mlp") m <- nnet(meta, y_ens, size=30, maxit=200, trace=F, MaxNWts = 50000)
        if(e$t=="rf") m <- randomForest(meta, y_ens, ntree=50)
        if(e$t=="svm") m <- svm(meta, y_ens, probability=T)
        models_crop[[e$n]] <- list(model=m, success=TRUE, type=paste0("ensemble_", e$t))
        cat(sprintf("✓ %s\n", e$n))
      }
    }
  }
  save(models_crop, file="trained_models/models_crop_pooling.RData")
  
  cat("\n¡PROCESO COMPLETADO EXITOSAMENTE!\n")
  unlink(TEMP_DIR, recursive = TRUE)
}

main()
