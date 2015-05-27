import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Vector;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.persist.EncogDirectoryPersistence;


public class MyClassifier implements IClassifier {
	
	public MyClassifier() {
		
	}
	
	/* IMPLEMENT THIS METHOD */
	@Override
	public String classify(int[] histogram) {
		return null;
	}
	
	/* IMPLEMENT THIS METHOD */
	@Override
	public void learn(Map<String, Vector<int[]>> dataSet) {
	}

}
