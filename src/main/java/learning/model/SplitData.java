package learning.model;

import java.util.ArrayList;
import java.util.List;

import weka.core.Instances;

public class SplitData{
	private List<Instances> trainingData = new ArrayList<Instances>();
	private List<Instances> testingData = new ArrayList<Instances>();
	
	public void addTrainingInstances(Instances trainingInstances){
		trainingData.add(trainingInstances);
	}
	
	public void addTestingInstances(Instances testingInstances){
		testingData.add(testingInstances);
	}

	public List<Instances> getTrainingData() {
		return trainingData;
	}

	public List<Instances> getTestingData() {
		return testingData;
	}
	
	
}
