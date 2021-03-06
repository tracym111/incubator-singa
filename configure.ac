#/**
# * Copyright 2015 The Apache Software Foundation
# *
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */
#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(singa, 0.1.0, dev@singa.incubator.apache.org)
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR(config)
AC_CONFIG_SRCDIR([src/utils/common.cc])
AC_CONFIG_HEADER([config.h])
AM_INIT_AUTOMAKE([subdir-objects foreign])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
ACLOCAL_AMFLAGS = -I m4 --install

# Checks for programs.
AC_PROG_CXX([g++])
AC_PROG_CC
AC_DISABLE_STATIC
AC_PROG_LIBTOOL

# Setup custom CUDA paths
AC_ARG_ENABLE(cuda,
	[AS_HELP_STRING(--enable-cuda,enable CUDA support)],
    cudaval="yes",
    cudaval="no")
AM_CONDITIONAL(DCUDA, [test "$cudaval" = "yes"])

AC_ARG_WITH([cuda],
   [AS_HELP_STRING(
        [--with-cuda=PATH],
        [prefix where CUDA is installed])],
   [cuda_prefix=$withval],
   [cuda_prefix="/usr/local/cuda"])
if test "$cuda_prefix" == "yes"; then
    if test "$withval" == "yes"; then
        cuda_prefix="/usr/local/cuda"
    fi
fi


if test x"$cudaval" = x"yes"; then
	AC_MSG_CHECKING([nvcc in $cuda_prefix/bin])
	if test -x "$cuda_prefix/bin/nvcc"; then
	    AC_MSG_RESULT([found])
	    AC_DEFINE_UNQUOTED([NVCC_PATH], ["$cuda_prefix/bin/nvcc"], [Path to nvcc binary])
	else
	    AC_MSG_RESULT([not found!])
	    AC_MSG_FAILURE([nvcc was not found in $cuda_prefix/bin])
	fi
    CUDA_CFLAGS="-I$cuda_prefix/include"
    CUDA_LDFLAGS="-L$cuda_prefix/lib64 -L$cuda_prefix/lib"
    CUDA_LIBS="-lcublas -lcudart -lcurand"
    NVCC="nvcc"
	DEBUG="-DUSE_GPU "
    AC_DEFINE(DCUDA,[1],[Defined if CUDA should be used])
	AC_CHECK_LIB([cuda], [main], [], [
 		AC_MSG_ERROR([unable to find cuda library])
 	 ])
	AC_CHECK_LIB([cudart], [main], [], [
	  	AC_MSG_ERROR([unable to find cudart library])
  	])
	AC_CHECK_LIB([curand], [main], [], [
 	 	AC_MSG_ERROR([unable to find curand library])
  	])
else
    CUDA_CFLAGS=""
    CUDA_LDFLAGS=""
    CUDA_LIBS=""
    NVCC=""
fi
AC_SUBST(NVCC)
AC_SUBST(CUDA_LDFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_CFLAGS)

# Checks for libraries.
AC_SEARCH_LIBS([cblas_sgemm], [openblas], [], [
  AC_MSG_ERROR([unable to find cblas_sgemm() function])
  ])
AC_SEARCH_LIBS([zmq_ctx_new], [zmq], [], [
  AC_MSG_ERROR([unable to find zmq_ctx_new() function])
    ])
AC_SEARCH_LIBS([zmsg_new], [czmq], [], [
  AC_MSG_ERROR([unable to find zmsg_new() function])
  ])
AC_CHECK_LIB([glog], [main], [], [
  AC_MSG_ERROR([unable to find glog library])
  ])
AC_CHECK_LIB([protobuf], [main], [], [
  AC_MSG_ERROR([unable to find protobuf library])
  ])

AC_ARG_ENABLE(lmdb,
     AS_HELP_STRING([--enable-lmdb],[enable debug option]),
	 [enable_lmdb=yes],[enable_lmdb=no])
AM_CONDITIONAL(LMDB, test "$enable_lmdb" = yes)
if test x"$enable_lmdb" = x"yes"; then
	AC_SEARCH_LIBS([mdb_env_create], [lmdb], [], [
	  AC_MSG_ERROR([unable to find mdb_env_create() function])
	  ])
	AC_DEFINE(LMDB, 1, [Enable Option layer])
fi

AC_ARG_ENABLE(test,
	AS_HELP_STRING([--enable-test],[enable singa test]),
	[enable_test=yes],[enable_test=no])
AM_CONDITIONAL(SINGATEST, test "$enable_test" = yes)
if test x"$enable_test" != x"no"; then
	PROGS='singatest test '
	LTLIBS='libgtest.la '
else
	PROGS=''
	LTLIBS=''
fi

AC_ARG_ENABLE(debug,
	AS_HELP_STRING([--enable-debug],[enable debug mode]),
	[enable_debug=yes],[enable_debug=no])
AM_CONDITIONAL(DEBUG, test "$enable_debug" = yes)
if test x"$enable_debug" != x"no"; then
	DEBUG+='-g'
else
	DEBUG+='-O2'
fi

AC_ARG_ENABLE(python,
    [AS_HELP_STRING([--enable-python],[enable python binding])],
    [enable_python=yes],[enable_python=no])
AM_CONDITIONAL([PY], [test "enable_python" = "yes"])
if test x"$enable_python" != x"no"; then
    AC_CHECK_PROG(PY_CHECK,python,yes)
    if test x"$PY_CHECK" != x"yes"; then
        AC_MSG_ERROR([Cannot find command "python". Please intall before make.])
    else    
        PYLIBS="-lpython`python -V 2>&1 | awk '{print substr($2,1,3)}'`"
    fi
fi        
AC_SUBST(PYLIBS)

AC_ARG_WITH([python],
    [AS_HELP_STRING([--with-python=PATH],[prefix where python is installed])],
    [python_prefix=$withval],[python_prefix="/usr/include/python`python -V 2>&1 | awk '{print substr($2,1,3)}'`"])
if test "$python_prefix" == "yes"; then
    if test "$withval" == "yes"; then
        python_prefix="/usr/include/python`python -V 2>&1 | awk '{print substr($2,1,3)}'`"
    fi
fi


if test x"$enable_python" != x"no"; then
    AC_MSG_CHECKING([Python.h in $python_prefix])
    if test -f "$python_prefix/Python.h"; then
        AC_MSG_RESULT([Python.h found])
        AC_DEFINE_UNQUOTED([PYTHON_PATH], ["$python_prefix"], [Path to python binary])
    else
        AC_MSG_RESULT([Python.h not found!])
        AC_MSG_FAILURE([Python.h was not found in $python_prefix])
    fi
    #AC_DEFINE(PY,[1],[Defined if PY should be used])
    PY_PROGS='_driver.la '
    PYFLAGS="-I$python_prefix "
else
    PY_PROGS=''
    PYFLAGS=''
fi

AC_SUBST([PROGS])
AC_SUBST([LTLIBS])
AC_SUBST([DEBUG])
AC_SUBST([PYFLAGS])
AC_SUBST([PY_PROGS])

#AC_CHECK_LIB([opencv_imgproc], [main], [], [
#  AC_MSG_ERROR([unable to find opencv_imgproc lib])
#  ])
#AC_CHECK_LIB([opencv_highgui], [main], [], [
#  AC_MSG_ERROR([unable to find opencv_highgui lib])
#  ])
#AC_CHECK_LIB([opencv_core], [main], [], [
#  AC_MSG_ERROR([unable to find opencv_core lib])
#  ])
AC_CHECK_LIB([zookeeper_mt], [main], [], [
  AC_MSG_ERROR([unable to find zookeeper])
  ])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([fcntl.h malloc.h stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_C_VOLATILE

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_STAT
AC_CHECK_FUNCS([gethostname memset mkdir pow sqrt])
AC_OUTPUT(Makefile)
