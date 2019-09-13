using System;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace FraudPredictionTrainer
{
    class Program
    {
        private static string DataPath = "data.csv";

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);

            var data = mlContext.Data.LoadFromTextFile<Transaction>(DataPath, hasHeader: true, separatorChar: ',');

            var testTrainData = mlContext.Data.TrainTestSplit(data);
            
            var dataProcessingPipeline = mlContext.Transforms.Categorical.OneHotEncoding("type")
                .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("nameDest"))
                .Append(mlContext.Transforms.Concatenate("Features", "type", "nameDest", "amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest")
                );
            
            var trainingPipeline = dataProcessingPipeline
                // .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "isFraud"));
                .Append(mlContext.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options 
                { 
                    NumberOfLeaves = 10, 
                    NumberOfTrees = 50,  
                    LabelColumnName = "isFraud", 
                    FeatureColumnName = "Features" 
                }));

            var trainedModel = trainingPipeline.Fit(testTrainData.TrainSet);
            
            var predictions = trainedModel.Transform(testTrainData.TestSet);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "isFraud");

            mlContext.Model.Save(trainedModel, data.Schema, "MLModel.zip");
        }
    }
}
