{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -w #-}
module Paths_MLLibrary (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where


import qualified Control.Exception as Exception
import qualified Data.List as List
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude


#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir `joinFileName` name)

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath



bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath
bindir     = "/workspaces/MLLibrary/main.hs/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/16a8ff52a7bea29abbb3542331f28c42f5dd9508c05cf0e2d29bad878c39caf7/9.2.8/bin"
libdir     = "/workspaces/MLLibrary/main.hs/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/16a8ff52a7bea29abbb3542331f28c42f5dd9508c05cf0e2d29bad878c39caf7/9.2.8/lib/x86_64-linux-ghc-9.2.8/MLLibrary-0.1.0.0-9o2NvuS9yGC1kzdUrCc1C5"
dynlibdir  = "/workspaces/MLLibrary/main.hs/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/16a8ff52a7bea29abbb3542331f28c42f5dd9508c05cf0e2d29bad878c39caf7/9.2.8/lib/x86_64-linux-ghc-9.2.8"
datadir    = "/workspaces/MLLibrary/main.hs/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/16a8ff52a7bea29abbb3542331f28c42f5dd9508c05cf0e2d29bad878c39caf7/9.2.8/share/x86_64-linux-ghc-9.2.8/MLLibrary-0.1.0.0"
libexecdir = "/workspaces/MLLibrary/main.hs/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/16a8ff52a7bea29abbb3542331f28c42f5dd9508c05cf0e2d29bad878c39caf7/9.2.8/libexec/x86_64-linux-ghc-9.2.8/MLLibrary-0.1.0.0"
sysconfdir = "/workspaces/MLLibrary/main.hs/.stack-work/install/x86_64-linux-tinfo6-libc6-pre232/16a8ff52a7bea29abbb3542331f28c42f5dd9508c05cf0e2d29bad878c39caf7/9.2.8/etc"

getBinDir     = catchIO (getEnv "MLLibrary_bindir")     (\_ -> return bindir)
getLibDir     = catchIO (getEnv "MLLibrary_libdir")     (\_ -> return libdir)
getDynLibDir  = catchIO (getEnv "MLLibrary_dynlibdir")  (\_ -> return dynlibdir)
getDataDir    = catchIO (getEnv "MLLibrary_datadir")    (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "MLLibrary_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "MLLibrary_sysconfdir") (\_ -> return sysconfdir)




joinFileName :: String -> String -> FilePath
joinFileName ""  fname = fname
joinFileName "." fname = fname
joinFileName dir ""    = dir
joinFileName dir fname
  | isPathSeparator (List.last dir) = dir ++ fname
  | otherwise                       = dir ++ pathSeparator : fname

pathSeparator :: Char
pathSeparator = '/'

isPathSeparator :: Char -> Bool
isPathSeparator c = c == '/'
