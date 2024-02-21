import weka.core.AttributeStats;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.Evaluation;

import javax.sound.sampled.Line;
import java.io.*;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("Height_Weight.csv"));
        Instances data = loader.getDataSet();

        int numInstances = data.numInstances();
        int numAttributes = data.numAttributes();

        //Question One
        for (int i = 0; i < numInstances; i++) {
            //double gender = data.instance(i).value(0);
            double heightInInches = data.instance(i).value(1);
            double weightInPounds = data.instance(i).value(2);

            //Convert height from inches to cm and weight from pounds to kilograms
            double heightInCm = heightInInches * 2.54;
            double weightInKg = weightInPounds * 0.453592;

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

        //Question 2
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

        //Question 3
        Randomize randomizeFilter = new Randomize();
        randomizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, randomizeFilter);
        data.setClassIndex(data.numAttributes() - 1);

        int splitIndex = (int) (data.numInstances() * 0.7);

        Instances trainingData = new Instances(data, 0, splitIndex);
        Instances testData = new Instances(data, splitIndex, data.numInstances() - splitIndex);

        //Question 4
        Instances subset = new Instances(trainingData, 0, Math.min(100, trainingData.numInstances()));
        LinearRegression model = new LinearRegression();
        model.buildClassifier(subset);

        Evaluation evaluation = new Evaluation(testData);
        evaluation.evaluateModel(model, testData);

        System.out.println("Regression Metrics for Model M1:");
        System.out.printf("Mean Absolute Error: %.4f", evaluation.meanAbsoluteError());
        System.out.println();
        System.out.printf("Root Mean Squared Error:  %.4f", evaluation.rootMeanSquaredError());
        System.out.println();
        System.out.printf("Relative Absolute Error:  %.4f%%", evaluation.relativeAbsoluteError());
        System.out.println();
        System.out.printf("Root Relative Squared Error:  %.4f%%", evaluation.rootRelativeSquaredError());
        System.out.println();
        System.out.printf("Correlation Coefficient:  %.4f", evaluation.correlationCoefficient());
        System.out.println();
        System.out.println("Number of Instances: " + subset.numInstances());


        //Question 5
        System.out.println();
        Instances subsetQuestion5 = new Instances(trainingData, 0, Math.min(1000, trainingData.numInstances()));
        LinearRegression modelTwo = new LinearRegression();
        modelTwo.buildClassifier(subsetQuestion5);

        Evaluation evaluationQuestion5 = new Evaluation(testData);
        evaluationQuestion5.evaluateModel(modelTwo, testData);

        System.out.println("Regression Metrics for Model M2:");
        System.out.printf("Mean Absolute Error: %.4f", evaluationQuestion5.meanAbsoluteError());
        System.out.println();
        System.out.printf("Root Mean Squared Error:  %.4f", evaluationQuestion5.rootMeanSquaredError());
        System.out.println();
        System.out.printf("Relative Absolute Error:  %.4f%%", evaluationQuestion5.relativeAbsoluteError());
        System.out.println();
        System.out.printf("Root Relative Squared Error:  %.4f%%", evaluationQuestion5.rootRelativeSquaredError());
        System.out.println();
        System.out.printf("Correlation Coefficient:  %.4f", evaluationQuestion5.correlationCoefficient());
        System.out.println();
        System.out.println("Number of Instances: " + subsetQuestion5.numInstances());

        //Question 6
        System.out.println();
        Instances subsetQuestion6 = new Instances(trainingData, 0, Math.min(5000, trainingData.numInstances()));
        LinearRegression modelThree = new LinearRegression();
        modelThree.buildClassifier(subsetQuestion6);

        Evaluation evaluationQuestion6 = new Evaluation(testData);
        evaluationQuestion6.evaluateModel(modelThree, testData);

        System.out.println("Regression Metrics for Model M3:");
        System.out.printf("Mean Absolute Error: %.4f", evaluationQuestion6.meanAbsoluteError());
        System.out.println();
        System.out.printf("Root Mean Squared Error:  %.4f", evaluationQuestion6.rootMeanSquaredError());
        System.out.println();
        System.out.printf("Relative Absolute Error:  %.4f%%", evaluationQuestion6.relativeAbsoluteError());
        System.out.println();
        System.out.printf("Root Relative Squared Error:  %.4f%%", evaluationQuestion6.rootRelativeSquaredError());
        System.out.println();
        System.out.printf("Correlation Coefficient:  %.4f", evaluationQuestion6.correlationCoefficient());
        System.out.println();
        System.out.println("Number of Instances: " + subsetQuestion6.numInstances());

        //Question 7
        System.out.println();
        Instances dataset = loader.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);
        LinearRegression modelFour = new LinearRegression();
        modelFour.buildClassifier(dataset);

        Evaluation evaluationQuestion7 = new Evaluation(dataset);
        evaluationQuestion7.evaluateModel(modelFour, dataset);

        System.out.println("Regression Metrics for Model M4:");
        System.out.printf("Mean Absolute Error: %.4f", evaluationQuestion7.meanAbsoluteError());
        System.out.println();
        System.out.printf("Root Mean Squared Error: %.4f", evaluationQuestion7.rootMeanSquaredError());
        System.out.println();
        System.out.printf("Relative Absolute Error:  %.4f%%", evaluationQuestion7.relativeAbsoluteError());
        System.out.println();
        System.out.printf("Root Relative Squared Error:  %.4f%%", evaluationQuestion7.rootRelativeSquaredError());
        System.out.println();
        System.out.printf("Correlation Coefficient: %.4f", evaluationQuestion7.correlationCoefficient());
        System.out.println();
        System.out.println("Number of Instances: " + dataset.numInstances());
        System.out.println();

        //Question 8
        System.out.println();
        System.out.println("First : Mean Absolute Error: ");
        System.out.println("In ascending order of largest Mean Absolute Error : M3= "+ evaluationQuestion6.meanAbsoluteError() + " M2= " + evaluationQuestion5.meanAbsoluteError()+
                " M1=" +evaluation.meanAbsoluteError() + " M4= " + evaluationQuestion7.meanAbsoluteError() + " which is M3 is indicating the smallest \n" +
                "average absolute difference between predicted and actual values, and M4 the largest average absolute difference between predicted and actual values."
        );

        System.out.println();
        System.out.println("Second: Root Mean Squared Error: ");
        System.out.println("In ascending order of Root Mean Squared Error : M3= "+ evaluationQuestion6.rootMeanSquaredError() + " M2= " + evaluationQuestion5.rootMeanSquaredError()+
                " M1=" +evaluation.rootMeanSquaredError() + " M4= " + evaluationQuestion7.rootMeanSquaredError() + " which is M3 is indicating better overall model accuracy, \n" +
                " and M4 the larger errors compared to the other models."
        );

        System.out.println();
        System.out.println("Third: Correlation Coefficient: ");
        System.out.println("M4 has a slightly lower coefficient = " + evaluationQuestion7.correlationCoefficient() + ", and M1, M2, and M3 have similar coefficients = " + evaluation.correlationCoefficient());


        System.out.println();
        System.out.println("Fourth: Relative Absolute Error: ");
        System.out.println("M3= " + evaluationQuestion6.relativeAbsoluteError()+ "% performs the best,then followed by M2= " + evaluationQuestion5.relativeAbsoluteError()+  "% M1=" + evaluation.relativeAbsoluteError() +  "% M4= " + evaluationQuestion7.relativeAbsoluteError()
                +"%. Lower values indicate better accuracy, and M3 have the smallest deviation from actual values. ");


        System.out.println();
        System.out.println("Fifth: Root Relative Squared Error: ");
        System.out.println(" M3 outperforms the other models, indicating superior overall model fit");

        System.out.println();
        System.out.println("sixth: Number of Instances: ");
        System.out.println("The Number of Instances increases from M1 to M4, reflecting the models' scalability to larger datasets. ");


        //Introduction
        System.out.println();
        LinearRegression linearRegression = new LinearRegression();
        linearRegression.buildClassifier(data);

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter a Person's Height (cm): ");
        double height = scanner.nextDouble();

        Instance instance = new DenseInstance(3);
        instance.setDataset(data);
        instance.setValue(1, height);

        double weight = linearRegression.classifyInstance(instance);
        System.out.println("A person that is " + height + "cm tall will be " + weight + " kilograms");

        scanner.close();
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