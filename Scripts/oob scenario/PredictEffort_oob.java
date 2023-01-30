import java.io.PrintWriter;
import java.io.File;
// import java.nio.charset.Charset;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
// import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.lmt.SimpleLinearRegression;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class PredictEffort_oob{

	static long trainingTime = 0;
	static long predictionTime = 0;
	static long runningTime = 0;
	static NumberFormat nf= new DecimalFormat("00.##");

	
	public static String dateFormatter(long millis) {
		long minutes = TimeUnit.MILLISECONDS.toMinutes(millis);
		double seconds = ((double)millis/1000)%60;
		return String.format("%02d:%05.2f",minutes, seconds);

	}

	
	public static <T> void printArray(T[] list) {
		StringBuffer buff = new StringBuffer();
		for(int i=0; i<list.length; i++) {
			if(i==0)
				buff.append("[");
			buff.append(list[i]);
			if(i==list.length-1)
				buff.append("]");
			else 
				buff.append(", ");
		}
		System.out.println(buff.toString());	
	}

	
	//Compute Mean Absolute Error
	public static double mae(double[] actuals, double[] preds) {
		int nInstances= preds.length;
		Stats stats= new Stats();
		for(int index=0; index<nInstances; index++) {
			stats.add(Math.abs(preds[index] - actuals[index]));
		}
		return stats.mean;
	}

	
	//Compute Median Absolute Error
	public static double medae(double[] actuals, double[] preds) {	
		int nInstances= preds.length;
		double[] ae = new double[nInstances];

		for(int index=0; index<nInstances; index++) {
			ae[index]= Math.abs(preds[index] - actuals[index]);
		}
		Arrays.sort(ae);
		if (ae.length%2==0) {
			return (ae[ae.length/2-1] + ae[ae.length/2])/2;
		} 
		return ae[ae.length/2];
	}

	
	//Compute Standard Deviation
	public static double sd(double[] actuals, double[] preds) {
		int nInstances= preds.length;
		Stats stats= new Stats();
		for(int index=0; index<nInstances; index++) {
			stats.add(Math.abs(preds[index] - actuals[index]));
		}
		stats.calculateDerived();
		return stats.stdDev;
	}

   // Removes the data columns that are not needed for prediction
	public static Instances cleanData(Instances data) throws Exception {
		Remove removeFilter = new Remove(); 
		// include the indices of the data columns below
		// removeFilter.setAttributeIndicesArray(new int[] {0,1,2,3,4,6});
		removeFilter.setAttributeIndicesArray(new int[] {0});
		removeFilter.setInputFormat(data);
		return new Instances(Filter.useFilter(data, removeFilter));		
	}

	
	public static Instances readData(String filename) throws Exception {
		ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename);
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		return cleanData(data);
	}

	
	public static Map<String, double[]> runModel(String modelName, Instances data) throws Exception {
		int ACTUAL_EFFORT_INDEX = data.classIndex();
		trainingTime = 0;
		predictionTime = 0;
		runningTime = 0;
		double[] predictions = new double[data.numInstances()];
		double[] y= data.attributeToDoubleArray(ACTUAL_EFFORT_INDEX); 
		//Splitting data according to LOO (Leave One Out)		
		for(int index=0; index<data.numInstances(); index++) {
			Instance testSet = data.instance(index);
			Instances trainSet = new Instances(data);
			trainSet.remove(index);
			
			if(modelName == "CART") {
				REPTree cartModel = new REPTree();
				long startTrainTime= System.currentTimeMillis();
				cartModel.buildClassifier(trainSet);
				long endTrainTime = System.currentTimeMillis(); 
				trainingTime += (endTrainTime-startTrainTime);
				double cartPredY = cartModel.classifyInstance(testSet);
				long endPredTime = System.currentTimeMillis();
				predictionTime += (endPredTime - endTrainTime);
				predictions[index]= cartPredY;
			}
			else if (modelName == "LR") {
				SimpleLinearRegression lrModel = new SimpleLinearRegression();
				long startTrainTime= System.currentTimeMillis();
				lrModel.buildClassifier(trainSet);
				long endTrainTime = System.currentTimeMillis();
				trainingTime += (endTrainTime-startTrainTime);
				double lrPredY= lrModel.classifyInstance(testSet);
				long endPredTime = System.currentTimeMillis();
				predictionTime += (endPredTime - endTrainTime);
				predictions[index]= lrPredY;
			}
			else if (modelName == "KNN") {
				IBk knnModel = new IBk();
				long startTrainTime= System.currentTimeMillis();
				knnModel.buildClassifier(trainSet);
				long endTrainTime = System.currentTimeMillis();
				trainingTime += (endTrainTime-startTrainTime);
				double knnPredY= knnModel.classifyInstance(testSet);
				long endPredTime = System.currentTimeMillis();
				predictionTime += (endPredTime - endTrainTime);
				predictions[index]= knnPredY;
			}
			else if (modelName == "SVM") {
				SMOreg svmModel = new SMOreg();
				long startTrainTime= System.currentTimeMillis();
				svmModel.buildClassifier(trainSet);
				long endTrainTime = System.currentTimeMillis();
				trainingTime += (endTrainTime-startTrainTime);
				double svmPredY= svmModel.classifyInstance(testSet);
				long endPredTime = System.currentTimeMillis();
				predictionTime += (endPredTime - endTrainTime);
				predictions[index]= svmPredY;
			}
		}
		runningTime = trainingTime + predictionTime;
		Map<String,double[]> results= new HashMap<String, double[]>();
		results.put("predictions", predictions);
		results.put("actuals", y);
		return results;
	}


	public static Double[] evaluateModel(double[] preds, double[] actuals) throws Exception {
		double maeValue= mae(actuals, preds);
		double medaeValue= medae(actuals, preds);
		double sdValue= sd(actuals, preds);

		return new Double[]{maeValue, medaeValue, sdValue};
	}	

	
	/***********************************************  RUN CODE  ******************************************************/
	public static void main(String[] args) {
		try {
			Instances data = readData("/Users/rebecca/Downloads/weka-example-master/data/ChinaComplete.csv");
			String result_path = "Weka_KitchDev_RQ1.csv";

			PrintWriter writer = new PrintWriter(new File(result_path));
			StringBuilder str= new StringBuilder();

			str.append(" ,MAE,MdAE,StDev,Time,TrainTime,PredTime\n");
			System.out.println("CART");
			for(int index=0; index<30; index++) {
				Map<String, double[]> cartPredictions= runModel("CART", data);
				Double[] cartResults= evaluateModel(cartPredictions.get("predictions"), cartPredictions.get("actuals"));
				str.append("CART,"+cartResults[0]+","+cartResults[1]+","+cartResults[2]+","+dateFormatter(runningTime)+","+dateFormatter(trainingTime)+","+dateFormatter(predictionTime)+"\n");
			}
			str.append(", , , , , ,\n");
			str.append(" ,MAE,MdAE,StDev,Time,TrainTime,PredTime\n");
			System.out.println("KNN");
			for(int index=0; index<30; index++) {
				Map<String, double[]> knnPredictions= runModel("KNN", data);
				Double[] knnResults= evaluateModel(knnPredictions.get("predictions"), knnPredictions.get("actuals"));	
				str.append("KNN,"+knnResults[0]+","+knnResults[1]+","+knnResults[2]+","+dateFormatter(runningTime)+","+dateFormatter(trainingTime)+","+dateFormatter(predictionTime)+"\n");
			}
			str.append(", , , , , ,\n");
			str.append(" ,MAE,MdAE,StDev,Time,TrainTime,PredTime\n");
			System.out.println("LR");
			for(int index=0; index<30; index++) {
				Map<String, double[]> lrPredictions= runModel("LR", data);
				Double[] lrResults= evaluateModel(lrPredictions.get("predictions"), lrPredictions.get("actuals"));
				str.append("LR,"+lrResults[0]+","+lrResults[1]+","+lrResults[2]+","+dateFormatter(runningTime)+","+dateFormatter(trainingTime)+","+dateFormatter(predictionTime)+"\n");
			}
			str.append(", , , , , ,\n");
			str.append(" ,MAE,MdAE,StDev,Time,TrainTime,PredTime\n");
			System.out.println("SVM");
			for(int index=0; index<30; index++) {
				Map<String, double[]> svmPredictions= runModel("SVM", data);
				Double[] svmResults= evaluateModel(svmPredictions.get("predictions"), svmPredictions.get("actuals"));
				str.append("SVM,"+svmResults[0]+","+svmResults[1]+","+svmResults[2]+","+dateFormatter(runningTime)+","+dateFormatter(trainingTime)+","+dateFormatter(predictionTime)+"\n");
			}
			writer.write(str.toString());
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
