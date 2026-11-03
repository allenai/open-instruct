# H161's image (dr layer, maxtext 1e23ebbc) plus fast resume: on resume the HF loader seeks each data host's row cursor
# to the position recorded at the nearest save (data_positions/<step>/process_<p>.json) instead of decoding and
# tokenizing every trained row again, then replays only the batches after that save. Training math is unchanged.
FROM us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3-dc88c6c7ea-dra35420de
COPY patches/ /tmp/patches/
RUN cd /deps && patch -p1 --forward --fuzz=0 < /tmp/patches/fastresume.patch
RUN python3 -c "import maxtext.input_pipeline.input_pipeline_utils as u, inspect; assert '_seek' in inspect.getsource(u.HFDataSource); print(u.__file__)" \
 && d=$(python3 -c "import maxtext, os; print(os.path.dirname(maxtext.__file__))") \
 && grep -q rows_consumed $d/input_pipeline/multihost_dataloading.py \
 && grep -q MAXTEXT_DATA_SKIP_ROWS $d/input_pipeline/hf_data_processing.py \
 && grep -q _find_position $d/trainers/post_train/hooks.py
