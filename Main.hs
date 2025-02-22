{-# LANGUAGE FlexibleContexts #-}

module Main where

import qualified Mllibrary as ML
import Numeric.LinearAlgebra

main :: IO ()
main = do
    -- Add your main program logic here
    -- Example:
    let xData = (2><3) [1,2,3,4,5,6] :: Matrix Double
        yData = vector [1,2] :: Vector Double
    
    -- Example usage of LinearRegression
    let model = ML.fitLinearRegression xData yData 0.01 1000
    putStrLn "Linear Regression Results:"
    print model
