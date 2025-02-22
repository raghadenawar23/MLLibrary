{-# LANGUAGE FlexibleContexts #-}

module Mllibrary where

import Data.List
import Numeric.LinearAlgebra
import System.Random (newStdGen, randomRs)
import Control.Monad (when)

--------------------------------------------------
-- Linear Regression
--------------------------------------------------

data LinearRegression = LinearRegression
  { lrWeights :: Vector Double,
    lrBias    :: Double
  }
  deriving (Show)

-- | Fit a Linear Regression model using gradient descent.
fitLinearRegression :: Matrix Double -> Vector Double -> Double -> Int -> LinearRegression
fitLinearRegression x y alpha nIters = go (konst 0 nFeatures) 0 0
  where
    (nSamples, nFeatures) = size x
    go :: Vector Double -> Double -> Int -> LinearRegression
    go w b iter
      | iter >= nIters = LinearRegression w b
      | otherwise =
          let predictions = (x #> w) + scalar b
              errors      = predictions - y
              dw          = scale (1 / fromIntegral nSamples) (tr x #> errors)
              db          = (1 / fromIntegral nSamples) * sumElements errors
              w'          = w - scale alpha dw
              b'          = b - alpha * db
          in go w' b' (iter + 1)

-- | Predict using a trained Linear Regression model.
predictLinearRegression :: LinearRegression -> Matrix Double -> Vector Double
predictLinearRegression (LinearRegression w b) x = (x #> w) + scalar b

--------------------------------------------------
-- Logistic Regression
--------------------------------------------------

data LogisticRegression = LogisticRegression
  { logWeights :: Vector Double,
    logBias    :: Double
  }
  deriving (Show)

-- Sigmoid activation function.
sigmoid :: Vector Double -> Vector Double
sigmoid z = 1 / (1 + cmap exp (scale (-1) z))

-- | Fit a Logistic Regression model using gradient descent.
fitLogisticRegression :: Matrix Double -> Vector Double -> Double -> Int -> LogisticRegression
fitLogisticRegression x y alpha nIters = go (konst 0 nFeatures) 0 0
  where
    (nSamples, nFeatures) = size x
    go :: Vector Double -> Double -> Int -> LogisticRegression
    go w b iter
      | iter >= nIters = LogisticRegression w b
      | otherwise =
          let z           = (x #> w) + scalar b
              predictions = sigmoid z
              errors      = predictions - y
              dw          = scale (1 / fromIntegral nSamples) (tr x #> errors)
              db          = (1 / fromIntegral nSamples) * sumElements errors
              w'          = w - scale alpha dw
              b'          = b - alpha * db
          in go w' b' (iter + 1)

-- | Predict binary labels (0 or 1) using a trained Logistic Regression model.
predictLogisticRegression :: LogisticRegression -> Matrix Double -> Vector Double
predictLogisticRegression (LogisticRegression w b) x =
  cmap (\p -> if p >= 0.5 then 1 else 0) $ sigmoid ((x #> w) + scalar b)

--------------------------------------------------
-- KMeans Clustering
--------------------------------------------------

-- | Compute Euclidean distance between two vectors.
euclideanDistance :: Vector Double -> Vector Double -> Double
euclideanDistance v1 v2 = sqrt $ sumElements $ cmap (** 2) (v1 - v2)

-- | Assign each row in the dataset to the closest centroid.
assignClusters :: Matrix Double -> [Vector Double] -> [Int]
assignClusters x centroids = map closest (toRows x)
  where
    closest row =
      snd $ minimum [(euclideanDistance row centroid, idx) | (centroid, idx) <- zip centroids [0 ..]]

-- | Compute new centroids as the mean of all points assigned to each cluster.
updateCentroids :: Matrix Double -> [Int] -> Int -> [Vector Double]
updateCentroids x clusters k =
  [ meanRows [row | (row, idx) <- zip (toRows x) clusters, idx == j]
    | j <- [0 .. k - 1]
  ]
  where
    meanRows [] = konst 0 (cols x)
    meanRows rs = scale (1 / fromIntegral (length rs)) (foldl1 (+) rs)

-- | Run the KMeans clustering algorithm.
kMeans :: Matrix Double -> Int -> Int -> IO ([Vector Double], [Int])
kMeans x k maxIters = do
    gen <- newStdGen
    let nSamples = rows x
        indices = take k $ randomRs (0, nSamples - 1) gen
        initialCentroids = [toRows x !! i | i <- indices]
        
        iterateK :: [Vector Double] -> Int -> IO [Vector Double]
        iterateK centroids iter 
            | iter >= maxIters = return centroids
            | otherwise = do
                let clusters = assignClusters x centroids
                    newCentroids = updateCentroids x clusters k
                if newCentroids == centroids
                    then return centroids
                    else iterateK newCentroids (iter + 1)
                    
    finalCentroids <- iterateK initialCentroids 0
    let finalClusters = assignClusters x finalCentroids
    return (finalCentroids, finalClusters)