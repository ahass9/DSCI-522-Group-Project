.PHONY: all dats figs models tables report clean-dats clean-figs clean-models clean-tables clean-all

all: dats figs models report

dats: \
	data/raw/hotel_data.csv \
	data/processed/hotel_data_cleaned.csv \
	data/processed/hotel_train_df.csv \
	data/processed/hotel_test_df.csv \
	data/processed/X_train_transformed.csv \
	data/processed/X_test_transformed.csv \
	data/processed/y_train.csv \
	data/processed/y_test.csv

data/raw/hotel_data.csv: scripts/download_data.py
	python scripts/download_data.py \
	--input-path="https://raw.githubusercontent.com/manthangandhi/hotel_cancellation_prediction/refs/heads/main/data/hotel_bookings.csv" \
	--output-path=data/raw/hotel_data.csv

data/processed/hotel_data_cleaned.csv: scripts/clean_data.py data/raw/hotel_data.csv
	python scripts/clean_data.py \
	--input-path=data/raw/hotel_data.csv \
	--output-path=data/processed/hotel_data_cleaned.csv

data/processed/hotel_train_df.csv \
data/processed/hotel_test_df.csv: \
	scripts/split_data.py \
	data/processed/hotel_data_cleaned.csv
	python scripts/split_data.py \
	--input-path=data/processed/hotel_data_cleaned.csv \
	--train-output-path=data/processed/hotel_train_df.csv \
	--test-output-path=data/processed/hotel_test_df.csv



figs: \
	results/figures/eda_correlation_matrix.png \
	results/figures/eda_deposit_type_vs_cancellations_count.png \
	results/figures/eda_hotel_vs_cancellations.png \
	results/figures/eda_lead_time_density.png \
	results/figures/eda_repeated_guest_vs_cancellations_count.png \
	results/figures/eda_reserved_room_type_vs_cancellations.png \
	results/figures/eda_target_var_distribution.png \
	results/figures/confusion_matrix_knn.png \
	results/figures/roc_curve_knn.png

results/figures/eda_correlation_matrix.png \
results/figures/eda_deposit_type_vs_cancellations_count.png \
results/figures/eda_hotel_vs_cancellations.png \
results/figures/eda_lead_time_density.png \
results/figures/eda_repeated_guest_vs_cancellations_count.png \
results/figures/eda_reserved_room_type_vs_cancellations.png \
results/figures/eda_target_var_distribution.png: \
	scripts/eda.py \
	data/processed/hotel_train_df.csv
	python scripts/eda.py \
	--input-path=data/processed/hotel_train_df.csv \
	--figure-prefix=results/figures/eda

models: \
	results/figures/confusion_matrix_knn.png \
	results/figures/roc_curve_knn.png \
	results/tables/test_accuracy.csv \
	results/tables/confusion_matrix.csv \
	results/models/preprocessor.pkl

data/processed/X_train_transformed.csv \
data/processed/X_test_transformed.csv \
data/processed/y_train.csv \
data/processed/y_test.csv \
results/models/preprocessor.pkl: \
	scripts/feature_preprocessing.py \
	data/processed/hotel_train_df.csv \
	data/processed/hotel_test_df.csv
	python scripts/feature_preprocessing.py \
	--train-input-path=data/processed/hotel_train_df.csv \
	--test-input-path=data/processed/hotel_test_df.csv \
	--x-train-transformed-output-path=data/processed/X_train_transformed.csv \
	--x-test-transformed-output-path=data/processed/X_test_transformed.csv \
	--y-train-output-path=data/processed/y_train.csv \
	--y-test-output-path=data/processed/y_test.csv \
	--preprocessor-output-path=results/models/preprocessor.pkl

results/figures/confusion_matrix_knn.png \
results/figures/roc_curve_knn.png \
results/tables/test_accuracy.csv \
results/tables/confusion_matrix.csv: \
	scripts/model_results.py \
	data/processed/X_train_transformed.csv \
	data/processed/X_test_transformed.csv \
	data/processed/y_train.csv \
	data/processed/y_test.csv
	python scripts/model_results.py \
	--x-train-path=data/processed/X_train_transformed.csv \
	--x-test-path=data/processed/X_test_transformed.csv \
	--y-train-path=data/processed/y_train.csv \
	--y-test-path=data/processed/y_test.csv \
	--metrics-output-path=results/tables/test_accuracy.csv \
	--cm-csv-output-path=results/tables/confusion_matrix.csv \
	--cm-figure-output-path=results/figures/confusion_matrix_knn.png \
	--roc-figure-output-path=results/figures/roc_curve_knn.png

tables: \
	results/tables/test_accuracy.csv \
	results/tables/confusion_matrix.csv

report: \
	reports/hotel_cancellation_classification_analysis.html \
	reports/hotel_cancellation_classification_analysis.pdf

reports/hotel_cancellation_classification_analysis.html \
reports/hotel_cancellation_classification_analysis.pdf: \
	reports/hotel_cancellation_classification_analysis.qmd \
	results/figures/confusion_matrix_knn.png \
	results/figures/roc_curve_knn.png \
	results/figures/eda_correlation_matrix.png \
	results/figures/eda_deposit_type_vs_cancellations_count.png \
	results/figures/eda_reserved_room_type_vs_cancellations.png \
	results/figures/eda_repeated_guest_vs_cancellations_count.png
	quarto render reports/hotel_cancellation_classification_analysis.qmd --to html
	quarto render reports/hotel_cancellation_classification_analysis.qmd --to pdf

clean-dats:
	rm -f data/raw/hotel_data.csv \
	data/processed/hotel_data_cleaned.csv \
	data/processed/hotel_train_df.csv \
	data/processed/hotel_test_df.csv \
	data/processed/X_train_transformed.csv \
	data/processed/X_test_transformed.csv \
	data/processed/y_train.csv \
	data/processed/y_test.csv

clean-figs:
	rm -f results/figures/*.png

clean-models:
	rm -f results/models/*.pkl

clean-tables:
	rm -f results/tables/*.csv

clean-all: clean-dats clean-figs clean-models clean-tables
	rm -f docs/reports/hotel_cancellation_classification_analysis.html \
	docs/reports/hotel_cancellation_classification_analysis.pdf 
	rm -rf reports/hotel_cancellation_classification_analysis_files







