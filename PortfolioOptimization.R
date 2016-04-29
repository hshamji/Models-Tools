#Sample portfolio optimizations - TMF
#optimize(result, interval = c(0,1), NewAsset = data[,3], OutputType ="FinalValue",maximum = TRUE)
#optimize(result, interval = c(0,1), NewAsset = data[,3], OutputType ="LastPeriodAverageValue",maximum = TRUE)
#optimize(result, interval = c(0,1), NewAsset = data[,3], OutputType ="Lifetime-AvgToTrough",maximum = FALSE)
#optimize(result, interval = c(0,1), NewAsset = data[,3], OutputType ="Lifetime-PeakToTrough",maximum = FALSE)


result <- function(NewAssetWeight=NULL, NewAsset=NULL, OutputType=NULL) {

  SPYData <- data[,1] #SPY price process picked up from global environment
  ExistingAssetWeight <- 1-NewAssetWeight 
  StartingCash <- 100
  RebalFreq <- 90 
  
  #To simplify the data manipulation we only retain n*RebalFreq data points, where n is an integer, trailing periods are discarded
  temp <- cbind(SPYData,NewAsset)
  temp <- temp[1:(RebalFreq * (nrow(temp)%/%RebalFreq))] 

  #Reshape Price Processes - each row of matrix starts after a rebalancing, we have a matrix of n rows, and RebalFreq columns
  PriceProcessA <- matrix(temp[,1], nrow= nrow(temp) %/% RebalFreq, ncol= RebalFreq, byrow=TRUE)
  PriceProcessB <- matrix(temp[,2], nrow= nrow(temp) %/% RebalFreq, ncol= RebalFreq, byrow=TRUE)
  
  #calculate asset % price change at each rebalancing
  PercentChangeA <- c(1,PriceProcessA[,1][-1] / PriceProcessA[,1][-nrow(PriceProcessA)])
  PercentChangeB <- c(1,PriceProcessB[,1][-1] / PriceProcessB[,1][-nrow(PriceProcessB)])
  
  #Use asset weights to calculate portfolio growth
  PortfolioGrowth <- PercentChangeA * ExistingAssetWeight + PercentChangeB * NewAssetWeight
  PortfolioValue <- StartingCash * cumprod(PortfolioGrowth)
  
  #Asset holdings are equal target weights at the rebalance period. Use this to back out asset holdings
  RebalHoldingsA <- PortfolioValue * ExistingAssetWeight
  RebalHoldingsB <- PortfolioValue * NewAssetWeight
  
  #Calculate intermediate values for assets based on asset price changes
  IntermediateChangeA <- PriceProcessA[,-1]/PriceProcessA[,-RebalFreq] # Cannot handle a daily rebalance
  IntermediateChangeB <- PriceProcessB[,-1]/PriceProcessB[,-RebalFreq] # Cannot handle a daily rebalance
  
  #Create matrix where first column is $value of asset, and subsequent columns are % changes
  CompleteHoldingsA <- cbind(RebalHoldingsA, IntermediateChangeA)
  CompleteHoldingsB<- cbind(RebalHoldingsB, IntermediateChangeB)
  
  #Cumulatively multiply row-wise to populate intermediate asset values
  for(i in 1:nrow(CompleteHoldingsA)) {CompleteHoldingsA[i,] <- cumprod(CompleteHoldingsA[i,])}
  for(j in 1:nrow(CompleteHoldingsB)) {CompleteHoldingsB[j,] <- cumprod(CompleteHoldingsB[j,])}
  
  #Populate objective output metrics (final values, rolling averages, etc.)
  Output <- temp
  Output$PortfolioValue <- c(t(CompleteHoldingsA) + t(CompleteHoldingsB))
  Output$RunMax <- runMax(Output$PortfolioValue, RebalFreq)
  Output$RunMin <- runMin(Output$PortfolioValue, RebalFreq)
  Output$RollMean <- rollmeanr(Output$PortfolioValue, RebalFreq)
  Output$RollPeakToTrough <- Output$RunMax / Output$RunMin
  Output$RollAvgToTrough <- Output$RollMean / Output$RunMin
  
  if (OutputType == "FinalValue") return(mean(tail(Output$PortfolioValue,n=1)))
  if (OutputType == "LastPeriodAverageValue") return(mean(tail(Output$PortfolioValue,n=RebalFreq), na.rm = TRUE))
  if (OutputType == "Lifetime-AvgToTrough") return(mean(Output$RollAvgToTrough, na.rm=TRUE))
  if (OutputType == "Lifetime-PeakToTrough") return(mean(Output$RollPeakToTrough, na.rm=TRUE))
  if (OutputType == "Matrix") return(Output)
}