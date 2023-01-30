package effortEstimation;

import java.io.File;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.KDTree;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.MathParameter;
import weka.core.setupgenerator.ParameterGroup;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.MultiSearch;
import weka.classifiers.meta.multisearch.DefaultEvaluationMetrics;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.lmt.SimpleLinearRegression;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class PredictEffort_tuning{

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
	// put the indices of the data columns below, for example: 
	// removeFilter.setAttributeIndicesArray(new int[] {0,1,2,3,4,6});
		removeFilter.setAttributeIndicesArray(new int[] {0});
		removeFilter.setInputFormat(data);
		return new Instances(Filter.useFilter(data, removeFilter));		
	}

	
	public static Instances readData(String filename) throws Exception {
		DataSource source = new DataSource(filename);
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
		
		MultiSearch multi = new MultiSearch();
		// SET 3-FOLDS CV
		DefaultSearch df= new DefaultSearch();
		df.setInitialSpaceNumFolds(3);
		multi.setAlgorithm(df);
		SelectedTag tag = new SelectedTag(DefaultEvaluationMetrics.EVALUATION_MAE, new DefaultEvaluationMetrics().getTags());
		multi.setEvaluation(tag);
		
		//Splitting data according to LOO (Leave One Out)		
		for(int index=0; index<data.numInstances(); index++) {
			Instance testSet = data.instance(index);
			Instances trainSet = new Instances(data);
			trainSet.remove(index);
			
			if(modelName == "CART") {
				REPTree cartModel = new REPTree();
				long startTrainTime= System.currentTimeMillis();
				// Try with maxDepth= 1,3,5,7,9,11,13,15,17,19,21
				MathParameter max_depth_value= new MathParameter();
				max_depth_value.setProperty("maxDepth");
				max_depth_value.setMin(1);
				max_depth_value.setMax(21);
				max_depth_value.setStep(2);
				max_depth_value.setExpression("I");
				ParameterGroup cart_params = new ParameterGroup();
				cart_params.setParameters(new AbstractParameter[]{max_depth_value});
			    multi.setClassifier(cartModel);
			    multi.setSearchParameters(new AbstractParameter[]{cart_params});  
				multi.buildClassifier(trainSet);				
				long endTrainTime = System.currentTimeMillis(); 
				trainingTime += (endTrainTime-startTrainTime);
				double cartPredY = multi.classifyInstance(testSet);
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
				KDTree kdTree= new KDTree();
				kdTree.setMaxInstInLeaf(30);
				knnModel.setNearestNeighbourSearchAlgorithm(kdTree);
				MathParameter k_value = new MathParameter();
				k_value.setProperty("KNN");
				//Try with k=1,5,9,13,17
				k_value.setMin(1);
				k_value.setMax(17);
				k_value.setStep(4);
				k_value.setExpression("I");
				ParameterGroup knn_params = new ParameterGroup();
				knn_params.setParameters(new AbstractParameter[]{k_value});
			    multi.setClassifier(knnModel);
			    multi.setSearchParameters(new AbstractParameter[]{knn_params});  
				multi.buildClassifier(trainSet);				
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
				RBFKernel rbfKern= new RBFKernel();
			    svmModel.setKernel(rbfKern);
			    svmModel.setRandomSeed(248);
				MathParameter c_value = new MathParameter();
                c_value.setProperty("c");
                //Try with C = from 0.25 to 4
                c_value.setMin(0.25);
                c_value.setMax(4);
                c_value.setStep(0.25);
                c_value.setExpression("I");
                MathParameter gamma_value = new MathParameter();
                gamma_value.setProperty("kernel.gamma");
                //Try with gamma= 0.1, 0.3, 0.5, 0.7, 0.9
                gamma_value.setMin(0.1);
                gamma_value.setMax(0.9);
                gamma_value.setStep(0.2);
                gamma_value.setExpression("I");
                ParameterGroup svm_params = new ParameterGroup();
                svm_params.setParameters(new AbstractParameter[]{c_value,gamma_value});
                multi.setClassifier(svmModel);
                multi.setSearchParameters(new AbstractParameter[]{svm_params});
				multi.buildClassifier(trainSet);
				long endTrainTime = System.currentTimeMillis();
				trainingTime += (endTrainTime-startTrainTime);
				double svmPredY= multi.classifyInstance(testSet);
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
			Instances data = readData("MiyazakiComplete.csv");
			String result_path = "Weka_Miyazaki_RQ1.csv";

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
