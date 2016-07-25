package learning.util;

import learning.model.SplitData;
import weka.core.Instances;

public class SplitterUtil {
	
	
	/**
	 * Splits all the original data up. Some will be used as training data. Some
	 * will be used as testing data.
	 * 
	 * @param data
	 * @param numberOfFolds
	 * @return
	 */
	public static SplitData crossValidationSplit(Instances data, int numberOfFolds) {
		SplitData splitData = new SplitData();

		for (int i = 0; i < numberOfFolds; i++) {
			splitData.addTrainingInstances(data.trainCV(numberOfFolds, i));
			splitData.addTestingInstances(data.testCV(numberOfFolds, i));
		}

		return splitData;
	}
}
