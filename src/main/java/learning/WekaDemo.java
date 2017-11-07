package learning;

import java.util.List;

import learning.model.SplitData;
import learning.util.AccuracyCalculator;
import learning.util.ClassificationUtil;
import learning.util.FileUtil;
import learning.util.SplitterUtil;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Tutorial based on:
 * http://www.programcreek.com/2013/01/a-simple-machine-learning-example-in-
 * java/
 * 
 * In this example, weka will attempt to predict the weather, based on the attributes given.
 * The use case here, is you have some data, and you want to classify it.
 * sunny = 0, overcast = 1, rainy = 2
 * 
 * 
 * 
 * 
 * 
 */
public class WekaDemo {
	private Classifier[] models = { 
			new J48(), // A decision tree
			new PART(), // A part decision list
			new DecisionTable(), // A decision table majority classifier
			new DecisionStump() // A one-level decision tree
	};

	public static void main(String[] args) throws Exception {
		new WekaDemo().findMostAccurateModel();
	}

	private void findMostAccurateModel() throws Exception {
		// Read the data from the file
		Instances data = FileUtil.getData("weather.txt", true);
		
		// Split up the data, some to be used as training data and some as testing data
		SplitData splitData = SplitterUtil.crossValidationSplit(data, 10);
		
		List<Instances> trainingData = splitData.getTrainingData();
		
		// For each type of model
		for (Classifier model: models) {
			FastVector predictions = new FastVector();
			for (int i = 0; i < trainingData.size(); i++) {
				Instances testData = splitData.getTestingData().get(i);
				
				// Make a prediction for the test data
				Evaluation validation = ClassificationUtil.classify(model, trainingData.get(i), testData);
				
				// Add it to the list of predictions
				predictions.appendElements(validation.predictions());
			}
			printClassificationAccuracy(model, AccuracyCalculator.calculateAccuracy(predictions));
			
			// Just showing here how you could do a prediction
			double prediction = model.classifyInstance(data.instance(10));
			System.out.println(prediction + " " + data.instance(10).classValue());
		}
	}

	private void printClassificationAccuracy(Classifier model, double accuracy) {
		System.out.println(model.getClass().getSimpleName() + " accuracy = " + String.format("%.2f%%", accuracy));		
	}
	
}