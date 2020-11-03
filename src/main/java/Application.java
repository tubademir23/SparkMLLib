
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
public class Application {
	static SparkSession sparkSession;
	public static void main(String[] args) {
		// LineerRegressionSample();
		NaiveBayesSample();
	}
	public static void NaiveBayesSample() {
		// features: hava,sicaklik,nem,ruzgar,basketbol
		sparkSession = SparkSession.builder().master("local")
				.appName("spark-mllib-naive-bayes").getOrCreate();

		Dataset<Row> raw_data = sparkSession.read().format("csv")
				.option("header", "true").option("inferSchema", "true")
				.load(".\\src\\data\\basketball.csv");
		// raw_data.show();
		StringIndexer indexHava = new StringIndexer().setInputCol("hava")
				.setOutputCol("hava_cat");
		StringIndexer indexSicaklik = new StringIndexer()
				.setInputCol("sicaklik").setOutputCol("sicaklik_cat");
		StringIndexer indexNem = new StringIndexer().setInputCol("nem")
				.setOutputCol("nem_cat");
		StringIndexer indexRuzgar = new StringIndexer().setInputCol("ruzgar")
				.setOutputCol("ruzgar_cat");
		// predicted column will be label not category
		StringIndexer indexBasketbol = new StringIndexer()
				.setInputCol("basketbol").setOutputCol("label");

		Dataset<Row> transformHavaData = indexHava.fit(raw_data)
				.transform(raw_data);
		Dataset<Row> transformSicaklikData = indexSicaklik
				.fit(transformHavaData).transform(transformHavaData);
		Dataset<Row> transformNemData = indexNem.fit(transformSicaklikData)
				.transform(transformSicaklikData);
		Dataset<Row> transformRuzgarData = indexRuzgar.fit(transformNemData)
				.transform(transformNemData);
		Dataset<Row> transformResultData = indexBasketbol
				.fit(transformRuzgarData).transform(transformRuzgarData);
		// transformResultData.show();

		VectorAssembler vectorAssembler = new VectorAssembler()
				.setInputCols(new String[]{"hava_cat", "sicaklik_cat",
						"nem_cat", "ruzgar_cat", "label"})
				.setOutputCol("features");
		Dataset<Row> transform = vectorAssembler.transform(transformResultData);
		// transform.show();
		Dataset<Row> final_data = transform.select("label", "features");
		// final_data.show();
		Dataset<Row>[] datasets = final_data
				.randomSplit(new double[]{0.7, 0.3});

		Dataset<Row> train_data = datasets[0];
		Dataset<Row> test_data = datasets[1];
		NaiveBayes nb = new NaiveBayes();
		nb.setSmoothing(1);
		NaiveBayesModel model = nb.fit(train_data);

		Dataset<Row> predictions = model.transform(test_data);
		predictions.show();

		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label").setPredictionCol("prediction")
				.setMetricName("accuracy");
		double evaluate = evaluator.evaluate(predictions);
		System.out.println("accuracy result" + evaluate);
	}
	public static void LineerRegressionSample() {
		sparkSession = SparkSession.builder().master("local")
				.appName("spark-mllib").getOrCreate();

		Dataset<Row> raw_data = sparkSession.read().format("csv")
				.option("header", "true").option("inferSchema", "true")
				.load(".\\src\\data\\sales.csv");

		Dataset<Row> new_data = sparkSession.read().format("csv")
				.option("header", "true").option("inferSchema", "true")
				.load(".\\src\\data\\test.csv");

		raw_data.show(40);;
		VectorAssembler features_vector = new VectorAssembler()
				.setInputCols(new String[]{"Ay"}).setOutputCol("features");

		Dataset<Row> transform = features_vector.transform(raw_data);

		Dataset<Row> transfor_new_data = features_vector.transform(new_data);

		Dataset<Row> final_data = transform.select("features", "Satis");

		Dataset<Row>[] datasets = final_data
				.randomSplit(new double[]{0.7, 0.3});

		// 0.7, 0.3 -> 23487.517
		// 0.9, 0.1 -> 25406.119
		Dataset<Row> train_data = datasets[0];
		Dataset<Row> test_data = datasets[1];

		LinearRegression lr = new LinearRegression();
		lr.setLabelCol("Satis");
		LinearRegressionModel model = lr.fit(train_data);

		// for excel: change code training test ratio to 1.0 and all transform
		// not est_data use train_data and get all prediction values.
		LinearRegressionTrainingSummary summary = model.summary();
		System.out.println(summary.r2());

		Dataset<Row> transform_test = model.transform(test_data);
		transform_test.show(30);

		Dataset<Row> transform_new_test = model.transform(transfor_new_data);
		transform_new_test.show();
	}

}
