### DOCUMENTATION.md

```markdown
# Haskell Machine Learning Library Documentation

## Overview

The Haskell Machine Learning Library is an educational project designed to introduce basic machine learning concepts and algorithms using Haskell. It implements three primary algorithms:

1. **Linear Regression:**  
   Utilizes gradient descent to fit a linear model to the data.
   
2. **Logistic Regression:**  
   Uses gradient descent along with a sigmoid activation function to perform binary classification.

3. **KMeans Clustering:**  
   Clusters data points into groups based on their Euclidean distance from cluster centroids.

This project is ideal for learners interested in both machine learning and functional programming with Haskell.

---

## Project Idea

The primary goal is to provide an accessible, well-documented example of how to implement and use basic machine learning algorithms in Haskell. By leveraging the [hmatrix](https://hackage.haskell.org/package/hmatrix) package, the library demonstrates efficient numerical computations and linear algebra operations, making it easier to understand the mathematical concepts behind the algorithms.

---

## Detailed Explanation of the Algorithms

### 1. Linear Regression

- **Concept:**  
  Linear regression models the relationship between a dependent variable and one or more independent variables by finding the best-fitting line through the data points.

- **Implementation Details:**  
  - The `fitLinearRegression` function uses gradient descent to minimize the error between predicted values and actual target values.
  - It iteratively updates the weights and bias based on the calculated gradients.

### 2. Logistic Regression

- **Concept:**  
  Logistic regression is used for binary classification. It maps the linear model output through the sigmoid function to produce probabilities, which are then converted to binary outcomes.

- **Implementation Details:**  
  - The `fitLogisticRegression` function employs gradient descent similar to linear regression.
  - It uses the sigmoid function to convert outputs into probabilities and applies a threshold (0.5) for classification.

### 3. KMeans Clustering

- **Concept:**  
  KMeans clustering is an unsupervised learning algorithm used to group data points into clusters based on similarity.

- **Implementation Details:**  
  - The algorithm starts by selecting random initial centroids.
  - It iteratively assigns each data point to the nearest centroid and updates centroids by computing the mean of assigned points.
  - The process stops when centroids no longer change or when the maximum number of iterations is reached.

---

## How to Use the Library

### Setup

1. **Install Prerequisites:**  
   Ensure that GHC and the `hmatrix` package are installed. For system dependencies, install BLAS/LAPACK as described in the README.

2. **Build the Project:**  
   Use either Cabal or Stack to build the project. Refer to the README for detailed instructions.

### Running Examples

- **Linear Regression Example:**  
  Modify and run the `Main.hs` file to experiment with linear regression. You can adjust parameters like learning rate, number of iterations, or input data as needed.

- **Testing Other Algorithms:**  
  While `Main.hs` demonstrates linear regression, you can extend it to test logistic regression or k-means clustering. Import the corresponding functions from `Mllibrary.hs` and incorporate them into your test code.

### Extending the Library

- **Adding New Algorithms:**  
  To add additional machine learning algorithms, create new functions within `Mllibrary.hs` or in separate modules.
- **Improving Existing Code:**  
  Contributions to optimize or refactor the code are welcome. Please update the documentation accordingly if you make any changes.

---

## Conclusion

This project serves as a learning tool for understanding both machine learning fundamentals and Haskell programming practices. Whether you are a student or an enthusiast, this library provides a solid foundation for further exploration and experimentation in functional programming and machine learning.

For further questions or contributions, please feel free to open an issue on GitHub.
```

---

### Main.hs

```haskell
{-# LANGUAGE FlexibleContexts #-}

module Main where

import qualified Mllibrary as ML
import Numeric.LinearAlgebra

main :: IO ()
main = do
    -- Example dataset: 2 samples with 3 features each
    let xData = (2><3) [1,2,3,4,5,6] :: Matrix Double
        yData = vector [1,2] :: Vector Double
    
    -- Train a Linear Regression model using gradient descent
    let model = ML.fitLinearRegression xData yData 0.01 1000
    putStrLn "Linear Regression Results:"
    print model
```

---

### Mllibrary.hs

```haskell
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
```

---
