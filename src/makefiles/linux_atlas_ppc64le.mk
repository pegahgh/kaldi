# ATLAS specific Linux ppc64le configuration

ifndef DOUBLE_PRECISION
$(error DOUBLE_PRECISION not defined.)
endif
ifndef OPENFSTINC
$(error OPENFSTINC not defined.)
endif
ifndef OPENFSTLIBS
$(error OPENFSTLIBS not defined.)
endif
ifndef ATLASINC
$(error ATLASINC not defined.)
endif
ifndef ATLASLIBS
$(error ATLASLIBS not defined.)
endif

CXXFLAGS = -std=c++11 -I.. -I$(OPENFSTINC) $(EXTRA_CXXFLAGS) \
           -Wall -Wno-sign-compare -Wno-unused-local-typedefs \
           -Wno-deprecated-declarations -Winit-self \
           -DKALDI_DOUBLEPRECISION=$(DOUBLE_PRECISION) \
           -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_ATLAS -I$(ATLASINC) \
           -m64 -maltivec -mcpu=power8 -mtune=power8 -mpower8-vector -mvsx \
           -pthread -rdynamic \
           -g # -O0 -DKALDI_PARANOID

ifeq ($(KALDI_FLAVOR), dynamic)
CXXFLAGS += -fPIC
endif

LDFLAGS = $(EXTRA_LDFLAGS) $(OPENFSTLDFLAGS) -rdynamic
LDLIBS = $(EXTRA_LDLIBS) $(OPENFSTLIBS) $(ATLASLIBS) -lm -lpthread -ldl
