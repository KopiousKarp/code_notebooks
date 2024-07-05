# Load libraries
library(ComplexHeatmap)
library(circlize)
library(dendextend)
library(reshape2)
library(ggplot2)
library(gridExtra)
# Load data
data <- read.csv("./n2_n3_degFC1_Up_down.csv", row.names = 1)
data <- data.matrix(data[, c("no_primordia", "primordia", "Emerge")])
# Scale data
scaled_data <- t(scale(t(data)))
# Create heatmap
heatmap <- Heatmap(scaled_data, name = "expression",
                   col = colorRamp2(c(-2, 0, 2), c("green", "black", "red")),
                   show_row_names = FALSE,
                   cluster_rows = TRUE,
                   cluster_columns = TRUE)
# Draw heatmap
draw(heatmap)
# Extract row clusters
row_dend <- row_dend(heatmap)
row_clusters <- cutree(as.hclust(row_dend), k = 4) # Change k as needed
# Add cluster information to data
data$cluster <- factor(row_clusters)
# Melt data for ggplot
data_melt <- melt(data, id.vars = "cluster")
# Line plots for each cluster
cluster_plots <- list()
for (cl in levels(data$cluster)) {
  print(paste("Processing cluster", cl))
  cluster_data <- data_melt[data_melt$cluster == cl, ]
  p <- ggplot(cluster_data, aes(variable, value, group = row.names(data), color = cluster)) +
    geom_line(alpha = 0.7) +
    ggtitle(paste("Group", cl, "(", nrow(data[data$cluster == cl, ]), "genes)", sep = " ")) +
    theme_minimal()
  cluster_plots[[cl]] <- p
}
# Arrange plots
grid.arrange(grobs = cluster_plots, ncol = 2)

