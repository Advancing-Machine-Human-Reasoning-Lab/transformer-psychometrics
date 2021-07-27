# Analyze data using Rasch model
# uses ltm package for IRT models


rasch_model <- function(data, cat, trial) {
	library(ltm)
    # dsc <- descript(data)
	# print(dsc)
 
    constrainedRasch <- rasch(data, constraint = cbind(length(data) + 1, 1))

	# graphing stuff
	fileName = paste("Rash Model: ", cat, " ", trial,".png")
	# print(fileName)
	
	# png(file=paste("Z:/Code/BERT-CDM/humanStudy/LM Psychometrics human study/",fileName))
	# ICC
	plot(constrainedRasch, legend = TRUE, cx = "bottomright", lwd = 3, cex.main = 1.5, cex.lab = 1.3, cex = 1.1)
	# test information
	# plot(constrainedRasch, type="IIC", items=0, lwd=3, cex.main=1.5, cex.lab=1.3)
	# item information curves
	# plot(constrainedRasch, type="IIC", annot=FALSE, lwd=3, cex.main=1.5, cex.lab=1.3)

	
	# plot(twoParameter, legend = TRUE, cx = "bottomright", lwd = 3, cex.main = 1.5, cex.lab = 1.3, cex = 1.1)
	# unConstrainedRasch <- rasch(data)
	# results_constrainedRasch <- summary(constrainedRasch)
	# results_unConstrainedRasch <- summary(unConstrainedRasch)
	gof_constrainedRasch <- GoF.rasch(constrainedRasch, B = 199)
	# gof_unConstrainedRasch <- GoF.rasch(unConstrainedRasch, B = 199)

	capture.output(print("Constrained Rasch Model:"),file="R_outputs.txt",append=TRUE)
	capture.output(summary(constrainedRasch),file="R_outputs.txt",append=TRUE)
	capture.output(gof_constrainedRasch,file="R_outputs.txt",append=TRUE)

}

# r.data <- data.frame(item_0 = c(1,0,0,0,0,0),item_2 = c(1,0,1,1,1,1))

# item_1 = c(1,1,1,1,1,1),