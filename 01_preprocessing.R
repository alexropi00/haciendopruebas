# ============================================================================
# 01_preprocessing.R - OPTIMIZADO PARA MÁXIMO USO DE CPU
# Preprocesamiento del dataset MNIST con dos alternativas:
# - Alternativa A: PCA con 95% de varianza explicada (MÁXIMO PARALELISMO)
# - Alternativa B: Crop + Average Pooling (CON EXPORTACIÓN DE FUNCIONES)
# ============================================================================

# Configurar hilos para operaciones matriciales ANTES de cargar librerías
n_threads <- parallel::detectCores()
Sys.setenv("OMP_NUM_THREADS" = n_threads)
Sys.setenv("MKL_NUM_THREADS" = n_threads)
Sys.setenv("OPENBLAS_NUM_THREADS" = n_threads)
Sys.setenv("VECLIB_MAXIMUM_THREADS" = n_threads)

cat(sprintf("Configurando %d hilos para operaciones matriciales\n", n_threads))

# Cargar librerías necesarias
library(parallel)
library(doParallel)
library(foreach)

# Configurar paralelización para foreach
n_cores <- max(1, detectCores() - 1)
cat(sprintf("Detectados %d núcleos, usando %d para foreach\n", detectCores(), n_cores))
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# ============================================================================
# FUNCIONES PARA LEER MNIST (archivos binarios IDX)
# ============================================================================

read_idx_images <- function(filepath) {
  con <- file(filepath, "rb")
  magic <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  n_images <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  n_rows <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  n_cols <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  
  cat(sprintf("  - Leyendo %d imágenes de %dx%d píxeles\n", n_images, n_rows, n_cols))
  
  n_pixels <- n_rows * n_cols
  images <- readBin(con, "integer", n = n_images * n_pixels, size = 1, signed = FALSE)
  close(con)
  
  images_matrix <- matrix(images, nrow = n_images, ncol = n_pixels, byrow = TRUE)
  return(images_matrix)
}

read_idx_labels <- function(filepath) {
  con <- file(filepath, "rb")
  magic <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  n_labels <- readBin(con, "integer", n = 1, size = 4, endian = "big")
  
  cat(sprintf("  - Leyendo %d etiquetas\n", n_labels))
  
  labels <- readBin(con, "integer", n = n_labels, size = 1, signed = FALSE)
  close(con)
  
  return(labels)
}

# Función para cargar MNIST
load_mnist_data <- function() {
  cat("Cargando datos MNIST...\n")
  
  base_path <- "./DataSetMINST/"
  
  train_images_file <- paste0(base_path, "train-images.idx3-ubyte")
  train_labels_file <- paste0(base_path, "train-labels.idx1-ubyte")
  test_images_file <- paste0(base_path, "t10k-images.idx3-ubyte")
  test_labels_file <- paste0(base_path, "t10k-labels.idx1-ubyte")
  
  cat("Train:\n")
  train_images <- read_idx_images(train_images_file)
  train_labels <- read_idx_labels(train_labels_file)
  
  cat("Test:\n")
  test_images <- read_idx_images(test_images_file)
  test_labels <- read_idx_labels(test_labels_file)
  
  cat(sprintf("\nTotal: %d imágenes train, %d imágenes test\n", 
              nrow(train_images), nrow(test_images)))
  
  return(list(
    train_images = train_images,
    train_labels = train_labels,
    test_images = test_images,
    test_labels = test_labels
  ))
}

# ============================================================================
# FUNCIONES AUXILIARES PARA PROCESAMIENTO DE IMÁGENES
# ============================================================================

vector_to_matrix <- function(img_vector) {
  matrix(img_vector, nrow = 28, ncol = 28, byrow = TRUE)
}

detect_bounding_box <- function(img_vector, threshold = 0) {
  img_mat <- vector_to_matrix(img_vector)
  
  rows_with_content <- which(rowSums(img_mat > threshold) > 0)
  cols_with_content <- which(colSums(img_mat > threshold) > 0)
  
  if (length(rows_with_content) == 0 || length(cols_with_content) == 0) {
    return(list(top = 1, bottom = 28, left = 1, right = 28,
                height = 28, width = 28))
  }
  
  top <- min(rows_with_content)
  bottom <- max(rows_with_content)
  left <- min(cols_with_content)
  right <- max(cols_with_content)
  
  return(list(
    top = top,
    bottom = bottom,
    left = left,
    right = right,
    height = bottom - top + 1,
    width = right - left + 1
  ))
}

crop_image_centered <- function(img_vector, target_height, target_width) {
  img_mat <- vector_to_matrix(img_vector)
  bbox <- detect_bounding_box(img_vector)
  
  digit_region <- img_mat[bbox$top:bbox$bottom, bbox$left:bbox$right]
  canvas <- matrix(0, nrow = target_height, ncol = target_width)
  
  digit_h <- nrow(digit_region)
  digit_w <- ncol(digit_region)
  
  start_row <- max(1, floor((target_height - digit_h) / 2) + 1)
  start_col <- max(1, floor((target_width - digit_w) / 2) + 1)
  
  end_row <- min(target_height, start_row + digit_h - 1)
  end_col <- min(target_width, start_col + digit_w - 1)
  
  digit_start_row <- 1
  digit_start_col <- 1
  digit_end_row <- digit_h
  digit_end_col <- digit_w
  
  if (digit_h > target_height) {
    digit_start_row <- floor((digit_h - target_height) / 2) + 1
    digit_end_row <- digit_start_row + target_height - 1
    start_row <- 1
    end_row <- target_height
  }
  
  if (digit_w > target_width) {
    digit_start_col <- floor((digit_w - target_width) / 2) + 1
    digit_end_col <- digit_start_col + target_width - 1
    start_col <- 1
    end_col <- target_width
  }
  
  canvas[start_row:end_row, start_col:end_col] <- 
    digit_region[digit_start_row:digit_end_row, digit_start_col:digit_end_col]
  
  return(as.vector(t(canvas)))
}

apply_average_pooling <- function(img_vector, current_size, pool_size = 2) {
  img_mat <- vector_to_matrix_any_size(img_vector, current_size)
  
  new_size <- floor(current_size / pool_size)
  pooled <- matrix(0, nrow = new_size, ncol = new_size)
  
  for (i in 1:new_size) {
    for (j in 1:new_size) {
      row_start <- (i - 1) * pool_size + 1
      row_end <- min(i * pool_size, current_size)
      col_start <- (j - 1) * pool_size + 1
      col_end <- min(j * pool_size, current_size)
      
      pooled[i, j] <- mean(img_mat[row_start:row_end, col_start:col_end])
    }
  }
  
  return(as.vector(t(pooled)))
}

vector_to_matrix_any_size <- function(img_vector, size) {
  matrix(img_vector, nrow = size, ncol = size, byrow = TRUE)
}

# ============================================================================
# ALTERNATIVA A: PCA TOTALMENTE PARALELO
# ============================================================================

preprocess_pca <- function(train_images, test_images, variance_threshold = 0.95) {
  cat("\n=== ALTERNATIVA A: PCA (MÁXIMO PARALELISMO) ===\n")
  cat("Aplicando PCA con 95% de varianza explicada...\n")
  
  n_rows <- nrow(train_images)
  n_cols <- ncol(train_images)
  
  cat("Paso 1: Normalizando datos en paralelo...\n")
  start_time <- Sys.time()
  
  chunk_size <- ceiling(n_rows / n_cores)
  row_chunks <- split(1:n_rows, ceiling(seq_along(1:n_rows) / chunk_size))
  
  train_norm_list <- foreach(chunk_idx = row_chunks, .combine = 'rbind') %dopar% {
    train_images[chunk_idx, ] / 255
  }
  train_norm <- train_norm_list
  
  test_norm <- test_images / 255
  
  end_time <- Sys.time()
  cat(sprintf("  Tiempo: %.2f segundos\n", 
              as.numeric(difftime(end_time, start_time, units = "secs"))))
  
  cat("Paso 2: Calculando media...\n")
  start_time <- Sys.time()
  train_mean <- colMeans(train_norm)
  end_time <- Sys.time()
  cat(sprintf("  Tiempo: %.2f segundos\n", 
              as.numeric(difftime(end_time, start_time, units = "secs"))))
  
  cat("Paso 3: Centrando datos en paralelo...\n")
  start_time <- Sys.time()
  
  train_centered_list <- foreach(chunk_idx = row_chunks, .combine = 'rbind') %dopar% {
    sweep(train_norm[chunk_idx, ], 2, train_mean, "-")
  }
  train_centered <- train_centered_list
  
  test_centered <- sweep(test_norm, 2, train_mean, "-")
  
  end_time <- Sys.time()
  cat(sprintf("  Tiempo: %.2f segundos\n", 
              as.numeric(difftime(end_time, start_time, units = "secs"))))
  
  cat("Paso 4: Calculando matriz de covarianza...\n")
  start_time <- Sys.time()
  
  cov_matrix <- crossprod(train_centered) / (n_rows - 1)
  
  end_time <- Sys.time()
  cat(sprintf("  Tiempo: %.2f segundos\n", 
              as.numeric(difftime(end_time, start_time, units = "secs"))))
  
  cat("Paso 5: Calculando eigenvalores y eigenvectores...\n")
  start_time <- Sys.time()
  
  eigen_result <- eigen(cov_matrix, symmetric = TRUE)
  
  end_time <- Sys.time()
  cat(sprintf("  Tiempo: %.2f segundos\n", 
              as.numeric(difftime(end_time, start_time, units = "secs"))))
  
  cat("Paso 6: Determinando número de componentes...\n")
  total_var <- sum(eigen_result$values)
  variance_explained <- cumsum(eigen_result$values) / total_var
  n_components <- which(variance_explained >= variance_threshold)[1]
  
  if (is.na(n_components)) {
    n_components <- length(eigen_result$values)
    cat("Advertencia: usando todos los componentes disponibles\n")
  }
  
  cat(sprintf("Número de componentes para %.1f%% de varianza: %d (de %d originales)\n",
              variance_threshold * 100, n_components, n_cols))
  cat(sprintf("Reducción dimensional: %.2f%%\n", 
              (1 - n_components / n_cols) * 100))
  
  cat("Paso 7: Proyectando datos en paralelo...\n")
  start_time <- Sys.time()
  
  rotation_matrix <- eigen_result$vectors[, 1:n_components]
  
  train_pca_list <- foreach(chunk_idx = row_chunks, .combine = 'rbind') %dopar% {
    train_centered[chunk_idx, ] %*% rotation_matrix
  }
  train_pca <- train_pca_list
  
  test_pca <- test_centered %*% rotation_matrix
  
  end_time <- Sys.time()
  cat(sprintf("  Tiempo: %.2f segundos\n", 
              as.numeric(difftime(end_time, start_time, units = "secs"))))
  
  return(list(
    train = train_pca,
    test = test_pca,
    rotation = rotation_matrix,
    n_components = n_components,
    train_mean = train_mean,
    eigenvalues = eigen_result$values[1:n_components]
  ))
}

# ============================================================================
# ALTERNATIVA B: Crop + Average Pooling (OPTIMIZADO)
# ============================================================================

preprocess_crop_pooling <- function(train_images, test_images, pool_size = 2) {
  cat("\n=== ALTERNATIVA B: CROP + POOLING ===\n")
  cat("Paso 1: Detectando bounding boxes de todas las imágenes...\n")
  
  all_images <- rbind(train_images, test_images)
  n_total <- nrow(all_images)
  
  # Exportar funciones necesarias a los workers
  clusterExport(cl, c("vector_to_matrix", "detect_bounding_box", 
                      "crop_image_centered", "apply_average_pooling", 
                      "vector_to_matrix_any_size"),
                envir = environment())
  
  cat("Procesando bounding boxes en paralelo...\n")
  start_time <- Sys.time()
  
  # Dividir trabajo en chunks más grandes para reducir overhead
  chunk_size <- ceiling(n_total / (n_cores * 4))
  image_chunks <- split(1:n_total, ceiling(seq_along(1:n_total) / chunk_size))
  
  bboxes <- foreach(chunk_idx = image_chunks, .combine = 'rbind') %dopar% {
    result <- matrix(0, nrow = length(chunk_idx), ncol = 2)
    for (i in seq_along(chunk_idx)) {
      bbox <- detect_bounding_box(all_images[chunk_idx[i], ])
      result[i, ] <- c(bbox$height, bbox$width)
    }
    result
  }
  
  end_time <- Sys.time()
  cat(sprintf("Tiempo detección bounding boxes: %.2f segundos\n", 
              as.numeric(difftime(end_time, start_time, units = "secs"))))
  
  max_height <- max(bboxes[, 1])
  max_width <- max(bboxes[, 2])
  
  cat(sprintf("Dimensiones máximas detectadas: %dx%d\n", max_height, max_width))
  cat(sprintf("Reducción por crop: de 28x28 (%d píxeles) a %dx%d (%d píxeles)\n",
              784, max_height, max_width, max_height * max_width))
  
  cat("Paso 2: Aplicando crop centrado en paralelo...\n")
  start_time <- Sys.time()
  
  cropped_images <- foreach(chunk_idx = image_chunks, .combine = 'rbind') %dopar% {
    result <- matrix(0, nrow = length(chunk_idx), ncol = max_height * max_width)
    for (i in seq_along(chunk_idx)) {
      result[i, ] <- crop_image_centered(all_images[chunk_idx[i], ], 
                                         max_height, max_width)
    }
    result
  }
  
  end_time <- Sys.time()
  cat(sprintf("Tiempo crop: %.2f segundos\n", 
              as.numeric(difftime(end_time, start_time, units = "secs"))))
  
  cat(sprintf("Paso 3: Aplicando average pooling (factor %d)...\n", pool_size))
  start_time <- Sys.time()
  
  pooled_size <- floor(max(max_height, max_width) / pool_size)
  max_dim <- max(max_height, max_width)
  
  pooled_images <- foreach(chunk_idx = image_chunks, .combine = 'rbind') %dopar% {
    result <- matrix(0, nrow = length(chunk_idx), ncol = pooled_size * pooled_size)
    for (i in seq_along(chunk_idx)) {
      result[i, ] <- apply_average_pooling(cropped_images[chunk_idx[i], ], 
                                           max_dim, pool_size)
    }
    result
  }
  
  end_time <- Sys.time()
  cat(sprintf("Tiempo pooling: %.2f segundos\n", 
              as.numeric(difftime(end_time, start_time, units = "secs"))))
  
  final_size <- pooled_size * pooled_size
  cat(sprintf("Dimensiones finales tras pooling: %dx%d (%d píxeles)\n",
              pooled_size, pooled_size, final_size))
  cat(sprintf("Reducción total: %.2f%%\n", 
              (1 - final_size / 784) * 100))
  
  pooled_images <- pooled_images / 255
  
  n_train <- nrow(train_images)
  train_processed <- pooled_images[1:n_train, ]
  test_processed <- pooled_images[(n_train + 1):n_total, ]
  
  return(list(
    train = train_processed,
    test = test_processed,
    crop_dimensions = c(max_height, max_width),
    pooled_size = pooled_size,
    pool_size = pool_size
  ))
}

# ============================================================================
# MAIN: Ejecución del preprocesamiento
# ============================================================================

main <- function() {
  total_start <- Sys.time()
  
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("INFORMACIÓN DEL SISTEMA\n")
  cat(rep("=", 70), "\n", sep = "")
  cat(sprintf("Núcleos detectados: %d\n", detectCores()))
  cat(sprintf("BLAS en uso: %s\n", ifelse(sessionInfo()$BLAS == "", 
                                          "R básico (monohilo)", 
                                          sessionInfo()$BLAS)))
  cat(rep("=", 70), "\n", sep = "")
  
  if (!dir.exists("processed_data")) {
    dir.create("processed_data")
  }
  
  mnist_data <- load_mnist_data()
  
  # ========================================
  # Alternativa A: PCA
  # ========================================
  pca_result <- preprocess_pca(
    mnist_data$train_images,
    mnist_data$test_images,
    variance_threshold = 0.95
  )
  
  cat("\nGuardando datos procesados con PCA...\n")
  train_labels <- mnist_data$train_labels
  test_labels <- mnist_data$test_labels
  save(
    pca_result,
    train_labels,
    test_labels,
    file = "processed_data/mnist_pca.RData"
  )
  
  # ========================================
  # Alternativa B: Crop + Pooling
  # ========================================
  crop_result <- preprocess_crop_pooling(
    mnist_data$train_images,
    mnist_data$test_images,
    pool_size = 2
  )
  
  cat("\nGuardando datos procesados con Crop+Pooling...\n")
  train_labels <- mnist_data$train_labels
  test_labels <- mnist_data$test_labels
  save(
    crop_result,
    train_labels,
    test_labels,
    file = "processed_data/mnist_crop_pooling.RData"
  )
  
  total_end <- Sys.time()
  
  cat("\n" , rep("=", 70), "\n", sep = "")
  cat("RESUMEN DEL PREPROCESAMIENTO\n")
  cat(rep("=", 70), "\n", sep = "")
  cat(sprintf("Tiempo total: %.2f minutos\n", 
              as.numeric(difftime(total_end, total_start, units = "mins"))))
  cat(sprintf("\nDatos originales: %d features\n", ncol(mnist_data$train_images)))
  cat(sprintf("\nAlternativa A (PCA):\n"))
  cat(sprintf("  - Features: %d\n", pca_result$n_components))
  cat(sprintf("  - Reducción: %.2f%%\n", 
              (1 - pca_result$n_components / 784) * 100))
  cat(sprintf("\nAlternativa B (Crop+Pooling):\n"))
  cat(sprintf("  - Features: %d\n", ncol(crop_result$train)))
  cat(sprintf("  - Reducción: %.2f%%\n", 
              (1 - ncol(crop_result$train) / 784) * 100))
  cat(sprintf("\nArchivos guardados en: processed_data/\n"))
  cat(rep("=", 70), "\n", sep = "")
  
  stopCluster(cl)
  
  cat("\n¡Preprocesamiento completado!\n")
}

# Ejecutar
main()
