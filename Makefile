nb2md:
	jupyter nbconvert \
	  --to markdown \
	  ./{노트북파일명}.ipynb \
	  --output {연}-{월}-{일}-{파일명}.md \
	  --output-dir . \
	  --ExtractOutputPreprocessor.enabled=True \
	  --ExtractOutputPreprocessor.output_filename_template="assets/images/posts/{연}-{월}-{일}/{cell_index}_{index}{extension}"
