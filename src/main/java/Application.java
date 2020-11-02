
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
public class Application {
	static SparkSession sparkSession;
	public static void main(String[] args) {
		sparkSession = SparkSession.builder().master("local")
				.appName("spark-mllib").getOrCreate();
		Dataset<Row> raw_data = sparkSession.read().format("csv")
				.option("header", "true").option("inferSchema", "true")
				.load(".\\src\\data\\sales.csv");

		Dataset<Row> new_data = sparkSession.read().format("csv")
				.option("header", "true").option("inferSchema", "true")
				.load(".\\src\\data\\test.csv");

		raw_data.show();
		VectorAssembler features_vector = new VectorAssembler()
				.setInputCols(new String[]{"Ay"}).setOutputCol("features");

		Dataset<Row> transform = features_vector.transform(raw_data);

		Dataset<Row> transfor_new_data = features_vector.transform(new_data);

		Dataset<Row> final_data = transform.select("features", "Satis");

		Dataset<Row>[] datasets = final_data
				.randomSplit(new double[]{0.9, 0.1});

		// 0.7, 0.3 -> 23487.517
		// 0.9, 0.1 -> 25406.119
		Dataset<Row> train_data = datasets[0];
		Dataset<Row> test_data = datasets[1];

		LinearRegression lr = new LinearRegression();
		lr.setLabelCol("Satis");
		LinearRegressionModel model = lr.fit(train_data);

		Dataset<Row> transform_test = model.transform(test_data);
		// transform_test.show();

		Dataset<Row> transform_new_test = model.transform(transfor_new_data);
		transform_new_test.show();

	}

}
