CXX ?= g++
EMCC ?= emcc

NATIVE_FLAGS := -O3 -march=native -mprefer-vector-width=512 -ffast-math -funroll-loops \
                -fopenmp-simd -std=c++17 -Wall -Wextra -pthread
WASM_FLAGS := -O3 -msimd128 -ffast-math -funroll-loops -fopenmp-simd -std=c++17 -Wall

.PHONY: all native wasm test bench clean serve

all: native

native: chad

chad: src/main.cpp src/vec3.h src/image.h src/scene.h src/trace.h src/render.h
	$(CXX) $(NATIVE_FLAGS) -o $@ src/main.cpp

WASM_EXPORTS := -sEXPORTED_FUNCTIONS=_malloc,_free,_chad_set_scene,_chad_scene_count,_chad_render,_chad_bench_primary,_chad_frame_ptr,_chad_stats_ptr,_chad_spheres_ptr,_chad_scene_info_ptr,_chad_lanes,_chad_selftest,_chad_load_mesh,_chad_has_mesh,_chad_mesh_verts_ptr,_chad_mesh_tris_ptr,_chad_grid_start_ptr,_chad_grid_start_len,_chad_grid_items_ptr,_chad_grid_items_len,_chad_grid_meta_ptr \
                -sEXPORTED_RUNTIME_METHODS=HEAPF32,HEAPU8,HEAPU32,HEAPF64

wasm: web/chad.js web/chad-st.js

web/chad.js: src/wasm.cpp src/vec3.h src/image.h src/scene.h src/trace.h src/render.h
	$(EMCC) $(WASM_FLAGS) -pthread src/wasm.cpp -o $@ \
	  -sMODULARIZE=1 -sEXPORT_NAME=createChad -sEXPORT_ES6=1 \
	  -sINITIAL_MEMORY=268435456 -sSTACK_SIZE=2097152 \
	  -sPTHREAD_POOL_SIZE='Math.min(32,(typeof navigator!=="undefined"&&navigator.hardwareConcurrency)||4)' \
	  $(WASM_EXPORTS) \
	  -sENVIRONMENT=web,worker,node \
	  -sALLOW_MEMORY_GROWTH=0

# Single-threaded fallback for browsers without cross-origin isolation.
web/chad-st.js: src/wasm.cpp src/vec3.h src/image.h src/scene.h src/trace.h src/render.h
	$(EMCC) $(WASM_FLAGS) src/wasm.cpp -o $@ \
	  -sMODULARIZE=1 -sEXPORT_NAME=createChad -sEXPORT_ES6=1 \
	  -sINITIAL_MEMORY=268435456 -sSTACK_SIZE=2097152 \
	  $(WASM_EXPORTS) \
	  -sENVIRONMENT=web,worker,node \
	  -sALLOW_MEMORY_GROWTH=0

test: chad
	./chad selftest

bench: chad
	./chad bench-all

serve:
	cd web && python3 -m http.server 8080

clean:
	rm -f chad web/chad.js web/chad.wasm out.png *.ppm
