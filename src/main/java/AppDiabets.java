import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class AppDiabets {
	static SparkSession sparkSession;
	public static void main(String[] args) {
		sparkSession = SparkSession.builder().master("local")
				.appName("spark-mllib").getOrCreate();
		Dataset<Row> raw_data = sparkSession.read().format("csv")
				.option("header", "true").option("inferSchema", "true")
				.load(".\\src\\data\\diabetes.csv");

		String[] headerList = {"Pregnancies", "Glucose", "BloodPressure",
				"SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
				"Age", "Outcome"};

		List<String> headers = Arrays.asList(headerList);
		List<String> headersResult = new ArrayList<String>();
		// result must be label other columns must be colname_cat pattern
		for (String h : headers) {
			StringIndexer indexTemp = new StringIndexer().setInputCol(h)
					.setOutputCol(h.equals("Outcome")
							? "label"
							: h.toLowerCase() + "_cat");
			raw_data = indexTemp.fit(raw_data).transform(raw_data);
			headersResult.add(
					h.equals("Outcome") ? "label" : h.toLowerCase() + "_cat");
		}
		// raw_data.show();

		VectorAssembler vectorAssembler = new VectorAssembler()
				.setInputCols(
						headersResult.toArray(new String[headersResult.size()]))
				.setOutputCol("features");
		Dataset<Row> transform = vectorAssembler.transform(raw_data); //
		// transform.show();
		Dataset<Row> final_data = transform.select("label", "features");
		final_data.show();
		Dataset<Row>[] datasets = final_data
				.randomSplit(new double[]{0.7, 0.3});

		// 0.7 result: 0.711
		// 0.8 result: 0.629
		// 0.6 result: 0.631
		// 0.9 result: 0.743
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

}
