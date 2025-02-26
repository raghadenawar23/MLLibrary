{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Mllibrary
    ( LinearRegression(..)
    , LogisticRegression(..)
    , fitLinearRegression
    , predictLinearRegression
    , fitLogisticRegression
    , predictLogisticRegression
    , kMeans
    ) where

import Numeric.LinearAlgebra
import System.Random

data LinearRegression = LinearRegression
    { weights :: Vector Double
    , bias    :: Double
    } deriving (Show)

-- | Fit a Linear Regression model using gradient descent.
fitLinearRegression :: Matrix Double -> Vector Double -> Double -> Int -> LinearRegression
fitLinearRegression x y alpha nIters = go (konst 0 nFeatures) 0 0
  where
    (nSamples, nFeatures) = size x
    go w b iter
      | iter >= nIters = LinearRegression w b
      | otherwise = go w' b' (iter + 1)
      where
        predictions = (x #> w) + scalar b
        error = predictions - y
        w' = w - scale (alpha / fromIntegral nSamples) (tr' x #> error)
        b' = b - (alpha / fromIntegral nSamples) * sumElements error

-- | Predict using a trained Linear Regression model.
predictLinearRegression :: LinearRegression -> Matrix Double -> Vector Double
predictLinearRegression (LinearRegression w b) x = (x #> w) + scalar b

data LogisticRegression = LogisticRegression
    { logWeights :: Vector Double
    , logBias    :: Double
    } deriving (Show)

sigmoid :: Vector Double -> Vector Double
sigmoid z = 1 / (1 + cmap exp (scale (-1) z))

fitLogisticRegression :: Matrix Double -> Vector Double -> Double -> Int -> LogisticRegression
fitLogisticRegression x y alpha nIters = go (konst 0 nFeatures) 0 0
  where
    (nSamples, nFeatures) = size x
    go w b iter
      | iter >= nIters = LogisticRegression w b
      | otherwise = go w' b' (iter + 1)
      where
        z = (x #> w) + scalar b
        predictions = sigmoid z
        error = predictions - y
        w' = w - scale (alpha / fromIntegral nSamples) (tr' x #> error)
        b' = b - (alpha / fromIntegral nSamples) * sumElements error

predictLogisticRegression :: LogisticRegression -> Matrix Double -> Vector Double
predictLogisticRegression (LogisticRegression w b) x =
    cmap (\p -> if p >= 0.5 then 1 else 0) $ sigmoid ((x #> w) + scalar b)

-- Helper functions for KMeans
euclideanDistance :: Vector Double -> Vector Double -> Double
euclideanDistance v1 v2 = sqrt $ sumElements $ cmap (** 2) (v1 - v2)

assignClusters :: Matrix Double -> [Vector Double] -> [Int]
assignClusters x centroids = map closest (toRows x)
  where
    closest row = snd $ minimum [(euclideanDistance row centroid, idx) | (centroid, idx) <- zip centroids [0..]]

updateCentroids :: Matrix Double -> [Int] -> Int -> [Vector Double]
updateCentroids x clusters k = 
    [meanRows [row | (row, idx) <- zip (toRows x) clusters, idx == j] | j <- [0..k-1]]
  where
    meanRows [] = konst 0 (cols x)
    meanRows rs = scale (1 / fromIntegral (length rs)) (foldl1 (+) rs)

kMeans :: Matrix Double -> Int -> Int -> IO ([Vector Double], [Int])
kMeans x k maxIters = do
    gen <- newStdGen
    let nSamples = rows x
        indices = take k $ randomRs (0, nSamples - 1) gen
        initialCentroids = [toRows x !! i | i <- indices]
        
        iterateK centroids iter
            | iter >= maxIters = return centroids
            | otherwise = do
                let clusters = assignClusters x centroids
                    newCentroids = updateCentroids x clusters k
                if all (\(c, nc) -> closeToZero (norm_2 (c - nc))) 
                       (zip centroids newCentroids)
                    then return newCentroids
                    else iterateK newCentroids (iter + 1)
        
        closeToZero val = abs val < 1e-9

    finalCentroids <- iterateK initialCentroids 0
    let finalClusters = assignClusters x finalCentroids
    return (finalCentroids, finalClusters)