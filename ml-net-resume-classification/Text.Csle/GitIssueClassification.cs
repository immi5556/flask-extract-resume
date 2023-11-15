using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Octokit;

namespace GitClass
{
    internal class GitIssueClassification
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        private static string BaseDatasetsRelativePath = @"./Data";
        private static string DataSetRelativePath = $"{BaseDatasetsRelativePath}/corefx-issues-train.tsv";
        private static string DataSetLocation = GetAbsolutePath(DataSetRelativePath);

        private static string BaseModelsRelativePath = @"./MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/GitHubLabelerModel.zip";
        private static string ModelPath = GetAbsolutePath(ModelRelativePath);


        public enum MyTrainerStrategy : int
        {
            SdcaMultiClassTrainer = 1,
            OVAAveragedPerceptronTrainer = 2
        };

        public static IConfiguration Configuration { get; set; }
        public static async Task Process()
        {
            SetupAppConfiguration();

            //1. ChainedBuilderExtensions and Train the model
            //BuildAndTrainModel(DataSetLocation, ModelPath, MyTrainerStrategy.OVAAveragedPerceptronTrainer);
            BuildAndTrainModel(DataSetLocation, ModelPath, MyTrainerStrategy.SdcaMultiClassTrainer);

            //2. Try/test to predict a label for a single hard-coded Issue
            TestSingleLabelPrediction(ModelPath);

            //3. Predict Issue Labels and apply into a real GitHub repo
            // (Comment the next line if no real access to GitHub repo) 
            //await PredictLabelsAndUpdateGitHub(ModelPath);

            ConsoleHelper.ConsolePressAnyKey();
        }

        public static void BuildAndTrainModel(string DataSetLocation, string ModelPath, MyTrainerStrategy selectedStrategy)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(DataSetLocation, hasHeader: true, separatorChar: '\t', allowSparse: false);

            // STEP 2: Common data process configuration with pipeline data transformations
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(GitHubIssue.Area))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "TitleFeaturized", inputColumnName: nameof(GitHubIssue.Title)))
                            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "DescriptionFeaturized", inputColumnName: nameof(GitHubIssue.Description)))
                            .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "TitleFeaturized", "DescriptionFeaturized"))
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
            var issue = new GitHubIssue() { ID = "Any-ID", Title = "WebSockets communication is slow in my machine", Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.." };
            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, GitHubIssuePrediction>(trainedModel);
            //Score
            var prediction = predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            //

            // STEP 6: Save/persist the trained model to a .ZIP file
            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);

            ConsoleHelper.ConsoleWriteHeader("Training process finalized");
        }

        private static void TestSingleLabelPrediction(string modelFilePathName)
        {
            var labeler = new Labeler(modelPath: ModelPath);
            labeler.TestPredictionForSingleIssue();
        }

        private static async Task PredictLabelsAndUpdateGitHub(string ModelPath)
        {
            Console.WriteLine(".............Retrieving Issues from GITHUB repo, predicting label/s and assigning predicted label/s......");

            var token = Configuration["GitHubToken"];
            var repoOwner = Configuration["GitHubRepoOwner"]; //IMPORTANT: This can be a GitHub User or a GitHub Organization
            var repoName = Configuration["GitHubRepoName"];

            if (string.IsNullOrEmpty(token) || token == "YOUR - GUID - GITHUB - TOKEN" ||
                string.IsNullOrEmpty(repoOwner) || repoOwner == "YOUR-REPO-USER-OWNER-OR-ORGANIZATION" ||
                string.IsNullOrEmpty(repoName) || repoName == "YOUR-REPO-SINGLE-NAME")
            {
                Console.Error.WriteLine();
                Console.Error.WriteLine("Error: please configure the credentials in the appsettings.json file");
                Console.ReadLine();
                return;
            }

            //This "Labeler" class could be used in a different End-User application (Web app, other console app, desktop app, etc.) 
            var labeler = new Labeler(ModelPath, repoOwner, repoName, token);

            await labeler.LabelAllNewIssuesInGitHubRepo();

            Console.WriteLine("Labeling completed");
            Console.ReadLine();
        }

        private static void SetupAppConfiguration()
        {
            var builder = new ConfigurationBuilder()
                                        .SetBasePath(Directory.GetCurrentDirectory())
                                        .AddJsonFile("appsettings.json");

            Configuration = builder.Build();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            var _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }

    internal class Labeler
    {
        private readonly GitHubClient _client;
        private readonly string _repoOwner;
        private readonly string _repoName;
        private readonly string _modelPath;
        private readonly MLContext _mlContext;

        private readonly PredictionEngine<GitHubIssue, GitHubIssuePrediction> _predEngine;
        private readonly ITransformer _trainedModel;

        private FullPrediction[] _fullPredictions;

        public Labeler(string modelPath, string repoOwner = "", string repoName = "", string accessToken = "")
        {
            _modelPath = modelPath;
            _repoOwner = repoOwner;
            _repoName = repoName;

            _mlContext = new MLContext();

            // Load model from file.
            _trainedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model.
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, GitHubIssuePrediction>(_trainedModel);

            // Configure Client to access a GitHub repo.
            if (accessToken != string.Empty)
            {
                var productInformation = new ProductHeaderValue("MLGitHubLabeler");
                _client = new GitHubClient(productInformation)
                {
                    Credentials = new Credentials(accessToken)
                };
            }
        }

        public void TestPredictionForSingleIssue()
        {
            var singleIssue = new GitHubIssue()
            {
                ID = "Any-ID",
                Title = "Crash in SqlConnection when using TransactionScope",
                Description = "I'm using SqlClient in netcoreapp2.0. Sqlclient.Close() crashes in Linux but works on Windows"
            };

            // Predict labels and scores for single hard-coded issue.
            var prediction = _predEngine.Predict(singleIssue);

            _fullPredictions = GetBestThreePredictions(prediction);

            Console.WriteLine($"==== Displaying prediction of Issue with Title = {singleIssue.Title} and Description = {singleIssue.Description} ====");

            Console.WriteLine("1st Label: " + _fullPredictions[0].PredictedLabel + " with score: " + _fullPredictions[0].Score);
            Console.WriteLine("2nd Label: " + _fullPredictions[1].PredictedLabel + " with score: " + _fullPredictions[1].Score);
            Console.WriteLine("3rd Label: " + _fullPredictions[2].PredictedLabel + " with score: " + _fullPredictions[2].Score);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
        }

        private FullPrediction[] GetBestThreePredictions(GitHubIssuePrediction prediction)
        {
            float[] scores = prediction.Score;
            int size = scores.Length;
            int index0, index1, index2 = 0;

            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            _predEngine.OutputSchema[nameof(GitHubIssuePrediction.Score)].GetSlotNames(ref slotNames);

            GetIndexesOfTopThreeScores(scores, size, out index0, out index1, out index2);

            _fullPredictions = new FullPrediction[]
                {
                    new FullPrediction(slotNames.GetItemOrDefault(index0).ToString(),scores[index0],index0),
                    new FullPrediction(slotNames.GetItemOrDefault(index1).ToString(),scores[index1],index1),
                    new FullPrediction(slotNames.GetItemOrDefault(index2).ToString(),scores[index2],index2)
                };

            return _fullPredictions;
        }

        private void GetIndexesOfTopThreeScores(float[] scores, int n, out int index0, out int index1, out int index2)
        {
            int i;
            float first, second, third;
            index0 = index1 = index2 = 0;
            if (n < 3)
            {
                Console.WriteLine("Invalid Input");
                return;
            }
            third = first = second = 000;
            for (i = 0; i < n; i++)
            {
                // If current element is  
                // smaller than first 
                if (scores[i] > first)
                {
                    third = second;
                    second = first;
                    first = scores[i];
                }
                // If arr[i] is in between first 
                // and second then update second 
                else if (scores[i] > second)
                {
                    third = second;
                    second = scores[i];
                }

                else if (scores[i] > third)
                    third = scores[i];
            }
            var scoresList = scores.ToList();
            index0 = scoresList.IndexOf(first);
            index1 = scoresList.IndexOf(second);
            index2 = scoresList.IndexOf(third);
        }

        // Label all issues that are not labeled yet
        public async Task LabelAllNewIssuesInGitHubRepo()
        {
            var newIssues = await GetNewIssues();
            foreach (var issue in newIssues.Where(issue => !issue.Labels.Any()))
            {
                var label = PredictLabels(issue);
                ApplyLabels(issue, label);
            }
        }

        private async Task<IReadOnlyList<Issue>> GetNewIssues()
        {
            var issueRequest = new RepositoryIssueRequest
            {
                State = ItemStateFilter.Open,
                Filter = IssueFilter.All,
                Since = DateTime.Now.AddMinutes(-10)
            };

            var allIssues = await _client.Issue.GetAllForRepository(_repoOwner, _repoName, issueRequest);

            // Filter out pull requests and issues that are older than minId
            return allIssues.Where(i => !i.HtmlUrl.Contains("/pull/"))
                            .ToList();
        }

        private FullPrediction[] PredictLabels(Octokit.Issue issue)
        {
            var corefxIssue = new GitHubIssue
            {
                ID = issue.Number.ToString(),
                Title = issue.Title,
                Description = issue.Body
            };

            _fullPredictions = Predict(corefxIssue);

            return _fullPredictions;
        }

        public FullPrediction[] Predict(GitHubIssue issue)
        {
            var prediction = _predEngine.Predict(issue);

            var fullPredictions = GetBestThreePredictions(prediction);

            return fullPredictions;
        }

        private void ApplyLabels(Issue issue, FullPrediction[] fullPredictions)
        {
            var issueUpdate = new IssueUpdate();

            //assign labels in GITHUB only if predicted score of all predictions is > 30%
            foreach (var fullPrediction in fullPredictions)
            {
                if (fullPrediction.Score >= 0.3)
                {
                    issueUpdate.AddLabel(fullPrediction.PredictedLabel);
                    _client.Issue.Update(_repoOwner, _repoName, issue.Number, issueUpdate);

                    Console.WriteLine($"Issue {issue.Number} : \"{issue.Title}\" \t was labeled as: {fullPredictions[0].PredictedLabel}");
                }
            }
        }
    }

    public class FullPrediction
    {
        public string PredictedLabel;
        public float Score;
        public int OriginalSchemaIndex;

        public FullPrediction(string predictedLabel, float score, int originalSchemaIndex)
        {
            PredictedLabel = predictedLabel;
            Score = score;
            OriginalSchemaIndex = originalSchemaIndex;
        }
    }

    internal class GitHubIssue
    {
        [LoadColumn(0)]
        public string ID;

        [LoadColumn(1)]
        public string Area; // This is an issue label, for example "area-System.Threading"

        [LoadColumn(2)]
        public string Title;

        [LoadColumn(3)]
        public string Description;
    }

    internal class GitHubIssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;

        public float[] Score;
    }

    internal class GitHubIssueTransformed
    {
        public string ID;
        public string Area;
        //public float[] Label;                 // -> Area dictionarized
        public string Title;
        //public float[] TitleFeaturized;       // -> Title Featurized 
        public string Description;
        //public float[] DescriptionFeaturized; // -> Description Featurized 
    }
}