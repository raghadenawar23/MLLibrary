# Project Structure

```
.
├── README.md
├── Main.hs
└── Mllibrary.hs
```

---

# README.md

```markdown
# Haskell Machine Learning Library

A simple machine learning library written in Haskell for educational purposes. This project includes implementations of:

- **Linear Regression** using gradient descent
- **Logistic Regression** for binary classification
- **KMeans Clustering** using Euclidean distance

The code leverages the [hmatrix](https://hackage.haskell.org/package/hmatrix) package for linear algebra computations.

## Project Structure

```
.
├── Main.hs          # Entry point demonstrating example usage of the library
├── Mllibrary.hs     # Module containing machine learning algorithms
└── README.md        # This file
```

## Prerequisites

- **GHC (Glasgow Haskell Compiler)**
  - Install via the [Haskell Platform](https://www.haskell.org/platform/) or [Stack](https://docs.haskellstack.org/).

- **hmatrix**
  - Provides necessary numerical and linear algebra functionality.

### System Dependencies

The `hmatrix` package depends on BLAS/LAPACK. Install them as follows:

- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install libblas-dev liblapack-dev
  ```
- **macOS (using Homebrew):**
  ```bash
  brew install openblas
  ```

## Installation

### Using Cabal

1. **Update Cabal Package List:**
   ```bash
   cabal update
   ```

2. **Install hmatrix:**
   ```bash
   cabal install hmatrix
   ```

3. **Build and Run:**
   If your project includes a `.cabal` file:
   ```bash
   cabal build
   cabal run
   ```

### Using Stack

1. **Create or Navigate to Your Stack Project:**
   ```bash
   stack new my-ml-project
   cd my-ml-project
   ```

2. **Configure Dependencies:**
   Add `hmatrix` to the `build-depends` section of your `.cabal` file or `package.yaml`:
   ```yaml
   build-depends:
     base >= 4.7 && < 5,
     hmatrix
   ```

3. **Build and Run:**
   ```bash
   stack build
   stack run
   ```

## Usage

The `Main.hs` file demonstrates how to use the library. For example, to run a linear regression:

```haskell
import qualified Mllibrary as ML
import Numeric.LinearAlgebra

main :: IO ()
main = do
    let xData = (2><3) [1,2,3,4,5,6] :: Matrix Double
        yData = vector [1,2] :: Vector Double
    let model = ML.fitLinearRegression xData yData 0.01 1000
    putStrLn "Linear Regression Results:"
    print model
```

Modify `Main.hs` to experiment with logistic regression or k-means clustering as needed.

## Contributing

Contributions, bug fixes, and improvements are welcome. Please fork this repository and submit a pull request with your changes.

## License

This project is provided for educational purposes. Choose an appropriate open-source license (e.g., MIT License) and include a corresponding license file.

## Acknowledgements

- The implementation leverages the [hmatrix](https://hackage.haskell.org/package/hmatrix) library for numerical computations.
- This project was created to help users understand basic machine learning algorithms using Haskell.
```

---

# Main.hs

```haskell
{-# LANGUAGE FlexibleContexts #-}

module Main where

import qualified Mllibrary as ML
import Numeric.LinearAlgebra

main :: IO ()
main = do
    -- Example dataset: 2 samples, 3 features
    let xData = (2><3) [1,2,3,4,5,6] :: Matrix Double
        yData = vector [1,2] :: Vector Double
    
    -- Example usage of Linear Regression
    let model = ML.fitLinearRegression xData yData 0.01 1000
    putStrLn "Linear Regression Results:"
    print model
```

---

# Mllibrary.hs

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

Once you have these files created in your repository, you can use GitHub to manage your project, build it with Cabal or Stack, and explore or extend the machine learning functionality as needed.
