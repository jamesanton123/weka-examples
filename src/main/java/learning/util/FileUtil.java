package learning.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;

import learning.WekaDemo;
import weka.core.Instances;

public class FileUtil {
	public static Instances getData(String resourceName, boolean useLastColumnAsTargetAttribute) throws IOException {
		BufferedReader datafile = FileUtil.readDataFile(resourceName);
		Instances instances = new Instances(datafile);
		// The class index indicates the target attribute used for classification.
		instances.setClassIndex(instances.numAttributes() - 1);
		return instances;
	}
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(
					new InputStreamReader(WekaDemo.class.getClassLoader().getResourceAsStream(filename), "UTF-8"));
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}

		return reader;
	}
}
