# Numerai Pipeline

My end-to-end Numerai prediction pipeline. Focus is staying accurate and keeping cloud costs down while still using a lot of AWS services. A lot of downsampling and room for offline training.

# Steps
~~1. Lambda train~~
~~2. Lambda make endpoint~~
~~3. Step function~~
~~4. Unique model name pass through step function~~
5. validate with pyspark
~~5. Lambda get live data~~
6. Predict with sagemaker endpoint
7. evaluate prediction
8. submit
9. retrain or not