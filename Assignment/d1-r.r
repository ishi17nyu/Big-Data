GDP_RAW <- read.csv(file="/home/data/MT/dataset1/gdp-1950-1983.csv",sep=",",head=TRUE)
GDP_RAW$X <- NULL
DF.TS <- ts(GDP_RAW[-1], start = 1950, frequency = 1)
jpeg('gdp_raw_plot.jpg')
plot(DF.TS, plot.type="single", pch = 1:ncol(DF.TS), col = 1:ncol(DF.TS))
legend("topleft", colnames(DF.TS), col=1:ncol(GDP_RAW), lty=1)
dev.off()

normalize <- function(x) {
    return ((x - mean(x)) / sd(x))
  }

GDP_SCALED <- as.data.frame(lapply(GDP_RAW[2:ncol(GDP_RAW)], normalize))
GDP_SCALED$X <- NULL
DF.TS <- ts(GDP_SCALED[-1], start = 1950, frequency = 1)
jpeg('gdp_scaled_plot.jpg')
plot(DF.TS, plot.type="single", pch = 1:ncol(DF.TS), col = 1:ncol(DF.TS))
legend("topleft", colnames(DF.TS), col=1:ncol(GDP_SCALED), lty=1)
dev.off()
