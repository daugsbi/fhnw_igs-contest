import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.Map.Entry;

public class KnnClassifier implements IClassifier {

	private final Map<Integer, Double> terms;
	private final Map<String, List<Doc>> training = new HashMap<>();

	public KnnClassifier(int k) {
		this.terms = new HashMap<>();
		for (int i = 0; i <= k; i++)
			this.terms.put(i, 0.0);
	}

	/* IMPLEMENT THIS METHOD */
	@Override
	public String classify(int[] histogram) {
		Doc d = new Doc(histogram, "");
		List<Doc> docs = new LinkedList<>();

		for (String k : training.keySet())
			docs.addAll(training.get(k));

		docs.add(d);
		calculateTfIdf(docs);

		Map<Doc, Double> classes = new HashMap<>();
		for (Doc x : docs) {
			classes.put(x, calculateSimilarity(x, d));
		}

		List<Entry<Doc, Double>> sorted = new LinkedList<>(classes.entrySet());
		Collections.sort(sorted, new Comparator<Entry<Doc, Double>>() {
			@Override
			public int compare(Entry<Doc, Double> o1, Entry<Doc, Double> o2) {
				return o2.getValue().compareTo(o1.getValue());
			}
		});
		
		Map<String, Integer> best = new HashMap<>();
		
		for(int i = 0; i < 15; i++) {
			String key = sorted.get(i).getKey().cls;
			if(!best.containsKey(key)) best.put(key, 0);
			best.put(key, best.get(key) + 1);
		}
		
		String winner = null;
		for(String key : best.keySet()) {
			if(winner == null) winner = key;
			else {
				if(best.get(winner) < best.get(key)) winner = key;
			}
		}
		
		return winner;
	}

	/* IMPLEMENT THIS METHOD */
	@Override
	public void learn(Map<String, Vector<int[]>> dataSet) {
		for (String key : dataSet.keySet()) {
			this.training.put(key, new LinkedList<>());
			for (int[] value : dataSet.get(key)) {
				this.training.get(key).add(new Doc(value, key));
			}
		}
	}

	private void calculateTfIdf(List<Doc> docs) {
		final int N = docs.size();

		// initialize empty "terms" map where a term
		// is the index inside the words array
		Map<Integer, Double> terms = new HashMap<>();
		for (Doc doc : docs) {
			for (int w : doc.words) {
				terms.put(w, 0.0);
			}
		}

		// go through each and calculate tf-idf for it
		for (Doc doc : docs) {
			doc.tfidfs = new HashMap<>(terms);

			// go through all "words" of this document
			for (int i = 0; i < doc.words.length; i++) {
				final int key = i;

				// calculate & normalize tf
				double tf = doc.words[i];
				tf = tf / (double) doc.words.length;

				// count in how much documents this word occurs
				long n = docs.stream().filter(d -> d.words[key] > 0).count();

				// calculate idf
				double idf = Math.log(N / (double) (n));

				// calculate tf*idf
				double tfidf = tf * idf;

				// check if tfidf is nan
				// this can happen when n is 0
				if (Double.isNaN(tfidf)) {
					tfidf = 0.0;
				}

				// write value back to word
				doc.tfidfs.put(i, tfidf);
			}
		}
	}

	private double calculateSimilarity(Doc d1, Doc d2) {
		Double v1[] = new Double[d1.tfidfs.size()];
		Double v2[] = new Double[d2.tfidfs.size()];
		d1.tfidfs.values().toArray(v1);
		d2.tfidfs.values().toArray(v2);

		// calculate similarity using cosine
		double dot = 0.0, root1 = 0.0, root2 = 0.0;
		for (int i = 0; i < v1.length; i++) {
			dot += v1[i] * v2[i];
			root1 += v1[i] * v1[i];
			root2 += v2[i] * v2[i];
		}
		double len1 = Math.sqrt(root1);
		double len2 = Math.sqrt(root2);

		// return result
		return dot / (len1 * len2);
	}

	private static class Doc {
		private int[] words;
		private Map<Integer, Double> tfidfs;
		private String cls;

		private Doc(int[] words, String cls) {
			this.cls = cls;
			this.words = words;
		}
	}

}
