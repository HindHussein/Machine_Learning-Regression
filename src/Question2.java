import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;

import java.io.*;

public class Question2 {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("Height_Weight.csv"));
        Instances data = loader.getDataSet();

        int numInstances = data.numInstances();
        int numAttributes = data.numAttributes();

        // Question One
        for (int i = 0; i < numInstances; i++) {
            //double gender = data.instance(i).value(0);
            double heightInInches = data.instance(i).value(1);
            double weightInPounds = data.instance(i).value(2);

            // Convert height and weight
            double heightInCm = heightInInches * 2.54;
            double weightInKg = weightInPounds * 0.453592;

            // Update the instances with the converted values
            data.instance(i).setValue(1, heightInCm);
            data.instance(i).setValue(2, weightInKg);

            System.out.println("Instance " + (i + 1) + ":");
            for (int j = 0; j < numAttributes; j++) {
                if(data.attribute(j).name().equalsIgnoreCase("Gender")){
                    if(data.instance(i).value(j)==1.0){
                        System.out.println("Attribute " + data.attribute(j).name() + ": Female");
                    }
                    else{
                        System.out.println("Attribute " + data.attribute(j).name() + ": Male");
                    }
                } else{
                    System.out.println("Attribute " + data.attribute(j).name() + ": " + data.instance(i).value(j));
                }

            }
            System.out.println();
        }

        // Question 2
        System.out.printf("%-20s%-12s%-12s%-12s%-12s%-12s\n", "Attribute", "Mean", "Median", "StdDev", "Min", "Max");

        for (int i = 1; i < numAttributes; i++) {
            AttributeStats attributeStats = data.attributeStats(i);

            double mean = attributeStats.numericStats.mean;
            double[] sortedValues = getSortedValues(data, i);
            double median = calculateMedian(sortedValues);
            double stdDev = attributeStats.numericStats.stdDev;
            double min = attributeStats.numericStats.min;
            double max = attributeStats.numericStats.max;

            System.out.printf("%-20s%-12.2f%-12.2f%-12.2f%-12.2f%-12.2f\n",
                    data.attribute(i).name(), mean, median, stdDev, min, max);


        }

        // Question 3
        // Randomize the order of instances
        Randomize randomizeFilter = new Randomize();
        randomizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, randomizeFilter);

        // Calculate the split index based on 70-30 split
        int splitIndex = (int) (data.numInstances() * 0.7);

        // Create training set
        Instances trainingData = new Instances(data, 0, splitIndex);

        // Create test set
        Instances testData = new Instances(data, splitIndex, data.numInstances() - splitIndex);



    }

    private static double[] getSortedValues(Instances data, int attributeIndex) {
        int numInstances = data.numInstances();
        double[] values = new double[numInstances];

        for (int i = 0; i < numInstances; i++) {
            values[i] = data.instance(i).value(attributeIndex);
        }

        java.util.Arrays.sort(values);

        return values;
    }

    private static double calculateMedian(double[] values) {
        int n = values.length;
        if (n % 2 == 0) {
            return (values[n/2 - 1] + values[n/2]) / 2.0;
        } else {
            return values[n/2];
        }
    }
}
