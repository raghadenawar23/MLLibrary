{-# LANGUAGE FlexibleContexts #-}

module Main where

import qualified Mllibrary as ML
import Numeric.LinearAlgebra
import Control.Exception (catch, SomeException)

main :: IO ()
main = catch runMain handleError
  where
    handleError :: SomeException -> IO ()
    handleError e = putStrLn $ "An error occurred: " ++ show e

runMain :: IO ()
runMain = do
    putStrLn "Starting ML Examples...\n"
    
    -- Linear Regression Example
    putStrLn "Running Linear Regression Example..."
    let x = (2 >< 3) [1, 2, 3,
                      4, 5, 6] :: Matrix Double
        y = vector [2.5, 6.0] :: Vector Double
        model = ML.fitLinearRegression x y 0.01 1000
        predictions = ML.predictLinearRegression model x

    putStrLn "Linear Regression Example:"
    putStrLn $ "Input Matrix X:\n" ++ show x
    putStrLn $ "Target Vector y:\n" ++ show y
    putStrLn $ "Predictions: " ++ show predictions

    -- Logistic Regression Example
    putStrLn "\nRunning Logistic Regression Example..."
    let x2 = (3 >< 2) [1, 2,
                       3, 4,
                       5, 6] :: Matrix Double
        y2 = vector [0, 0, 1] :: Vector Double
        logModel = ML.fitLogisticRegression x2 y2 0.01 1000
        logPredictions = ML.predictLogisticRegression logModel x2

    putStrLn "\nLogistic Regression Example:"
    putStrLn $ "Input Matrix X2:\n" ++ show x2
    putStrLn $ "Target Vector y2:\n" ++ show y2
    putStrLn $ "Predictions: " ++ show logPredictions

    -- KMeans Example
    putStrLn "\nRunning KMeans Clustering Example..."
    let x3 = (4 >< 2) [1, 2,
                       8, 7,
                       2, 1,
                       7, 8] :: Matrix Double
    (centroids, clusters) <- ML.kMeans x3 2 100

    putStrLn "\nKMeans Clustering Example:"
    putStrLn $ "Input Matrix X3:\n" ++ show x3
    putStrLn $ "Clusters: " ++ show clusters
    putStrLn $ "Centroids: " ++ show centroids