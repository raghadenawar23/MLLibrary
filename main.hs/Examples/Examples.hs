module Main where

import Mllibrary
import Numeric.LinearAlgebra

main :: IO ()
main = do
    -- Linear Regression Example
    let x = (2 >< 3) [1, 2, 3,
                      4, 5, 6] :: Matrix Double
        y = vector [2.5, 6.0] :: Vector Double
        model = fitLinearRegression x y 0.01 1000
        predictions = predictLinearRegression model x
    
    putStrLn "Linear Regression Example:"
    putStrLn $ "Predictions: " ++ show predictions
    
    -- Logistic Regression Example
    let x2 = (3 >< 2) [1, 2,
                       3, 4,
                       5, 6] :: Matrix Double
        y2 = vector [0, 0, 1] :: Vector Double
        logModel = fitLogisticRegression x2 y2 0.01 1000
        logPredictions = predictLogisticRegression logModel x2
    
    putStrLn "\nLogistic Regression Example:"
    putStrLn $ "Predictions: " ++ show logPredictions
    
    -- KMeans Example
    let x3 = (4 >< 2) [1, 2,
                       8, 7,
                       2, 1,
                       7, 8] :: Matrix Double
    (centroids, clusters) <- kMeans x3 2 100
    
    putStrLn "\nKMeans Clustering Example:"
    putStrLn $ "Clusters: " ++ show clusters
    putStrLn $ "Centroids: " ++ show centroids
