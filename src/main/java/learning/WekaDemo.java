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
import weka.core.Instances;

/**
 * Tutorial based on:
 * http://www.programcreek.com/2013/01/a-simple-machine-learning-example-in-
 * java/
 */
public class WekaDemo {
	private Classifier[] models = { 
			new J48(), // A decision tree
			new PART(), // A part decision list
			new DecisionTable(), // A decision table majority classifier
			new DecisionStump() // A one-level decision tree
	};

	public static void main(String[] args) throws Exception {
		new WekaDemo().runDemo();
	}

	private void runDemo() throws Exception {
		Instances data = FileUtil.getData("weather.txt", true);		
		SplitData splitData = SplitterUtil.crossValidationSplit(data, 10);
		List<Instances> trainingData = splitData.getTrainingData();
		List<Instances> testingData = splitData.getTestingData();		
		for (Classifier model: models) {
			FastVector predictions = new FastVector();
			for (int i = 0; i < trainingData.size(); i++) {
				Evaluation validation = ClassificationUtil.classify(model, trainingData.get(i), testingData.get(i));
				predictions.appendElements(validation.predictions());
			}
			printClassificationAccuracy(model, AccuracyCalculator.calculateAccuracy(predictions));
		}
	}

	private void printClassificationAccuracy(Classifier model, double accuracy) {
		System.out.println(model.getClass().getSimpleName() + " accuracy = " + String.format("%.2f%%", accuracy));		
	}
	
}