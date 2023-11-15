using GitClass;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static GitClass.GitIssueClassification;

namespace Text.Csle
{
    internal class ResumeCategory
    {
        private static string BaseDatasetsRelativePath = @"./Data";
        private static string DataSetRelativePath = $"{BaseDatasetsRelativePath}/jobss.csv";
        private static string DataSetLocation = GetAbsolutePath(DataSetRelativePath);

        private static string BaseModelsRelativePath = @"./MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/Jobss_v1_Model.zip";
        private static string ModelPath = GetAbsolutePath(ModelRelativePath);
        public static void PredictProcess()
        {
            var mlContext = new MLContext(seed: 1);
            DataViewSchema modelSchema;
            ITransformer trainedModel = mlContext.Model.Load("./MLModels/Jobss_v1_Model.zip", out modelSchema);
            var issue = new ResumeLabel() { KeySkill = "WebSockets communication is slow in my machine", RoleCategory = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.." };
            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<ResumeLabel, ResumePrediction>(trainedModel);
            //Score
            var prediction = predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.JobTitle}, Score: {prediction.Score}  ===============");
        }
        public static void BuildAndTrainModel(MyTrainerStrategy selectedStrategy)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<ResumeLabel>(DataSetLocation, hasHeader: true, separatorChar: ',', allowSparse: false);
            Console.WriteLine(trainingDataView.GetRowCount());
            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(ResumeLabel.JobTitle))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "KeySkillFeaturized", inputColumnName: nameof(ResumeLabel.KeySkill)))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "RoleCategoryFeaturized", inputColumnName: nameof(ResumeLabel.RoleCategory)))
                            .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "KeySkillFeaturized", "RoleCategoryFeaturized"))
                            .AppendCacheCheckpoint(mlContext);
            // Use in-memory cache for small/medium datasets to lower training time. 
            // Do NOT use it (remove .AppendCacheCheckpoint()) when handling very large datasets.

            // (OPTIONAL) Peek data (such as 2 records) in training DataView after applying the ProcessPipeline's transformations into "Features" 
            ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 2);

            // STEP 3: Create the selected training algorithm/trainer
            IEstimator<ITransformer> trainer = null;
            switch (selectedStrategy)
            {
                case MyTrainerStrategy.SdcaMultiClassTrainer:
                    trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");
                    break;
                case MyTrainerStrategy.OVAAveragedPerceptronTrainer:
                    {
                        // Create a binary classification trainer.
                        var averagedPerceptronBinaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10);
                        // Compose an OVA (One-Versus-All) trainer with the BinaryTrainer.
                        // In this strategy, a binary classification algorithm is used to train one classifier for each class, "
                        // which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers, "
                        // and choosing the prediction with the highest confidence score.
                        trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer);

                        break;
                    }
                default:
                    break;
            }

            //Set the trainer/algorithm and map label to value (original readable state)
            var trainingPipeline = dataProcessPipeline.Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Cross-Validate with single dataset (since we don't have two datasets, one for training and for evaluate)
            // in order to evaluate and get the model's accuracy metrics

            Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
            var crossValidationResults = mlContext.MulticlassClassification.CrossValidate(data: trainingDataView, estimator: trainingPipeline, numberOfFolds: 6, labelColumnName: "Label");

            ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

            // STEP 5: Train the model fitting to the DataSet
            Console.WriteLine("=============== Training the model ===============");
            var trainedModel = trainingPipeline.Fit(trainingDataView);

            // (OPTIONAL) Try/test a single prediction with the "just-trained model" (Before saving the model)
            var issue = new ResumeLabel() { KeySkill = "WebSockets communication is slow in my machine", RoleCategory = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.." };
            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<ResumeLabel, ResumePrediction>(trainedModel);
            //Score
            var prediction = predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.JobTitle}, Score: {prediction.Score}  ===============");
            //

            // STEP 6: Save/persist the trained model to a .ZIP file
            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            ConsoleHelper.ConsoleWriteHeader("Training process finalized");
        }
    }

    internal class ResumeLabel
    {
        [LoadColumn(0)]
        public string JobTitle;

        [LoadColumn(2)]
        public string JobExp; // This is an issue label, for example "area-System.Threading"

        [LoadColumn(3)]
        public string KeySkill;

        [LoadColumn(4)]
        public string RoleCategory;

        [LoadColumn(6)]
        public string FunctionalArea;
        [LoadColumn(7)]
        public string Industry;
        [LoadColumn(8)]
        public string Role;
        [LoadColumn(11)]
        public string Salary;
    }
    internal class ResumePrediction
    {
        [ColumnName("PredictedLabel")]
        public string JobTitle;

        public float[] Score;
    }
}
