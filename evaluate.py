if __name__ == "__main__":
  
    # model_dir_ViT = r"runs\384_Strong_Aug\vit_small_patch16_224_in21k\LR_0.01"
    # model_dir_BiT = r"runs\384_Strong_Aug\resnetv2_50x1_bitm_in21k\LR_0.01"
    # ensemble_dir = r'runs\ensemble'
    
    # # Load datasets split into train, val and test
    # print(sys.argv)
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "data/eyePACs"
    # model_directory = sys.argv[2] if len(sys.argv) > 2 else r"runs\384_Run_Baseline\vit_small_patch16_224_in21k_eyePACS_LR_0.01"
    # model_name = sys.argv[3] if len(sys.argv) > 3 else "vit_small_patch16_224_in21k"#"resnetv2_50x1_bitm_in21k" vit_small_patch16_224_in21k
    # phase = sys.argv[4] if len(sys.argv) > 4 else "val"

    # dataset_proportions = np.array([0.6, 0.2, 0.2])
    # full_dataset = EyePACS_Dataset(data_directory, random_state=13, img_size=384)
    # class_names = full_dataset.class_names
    # datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, ["train", "val", "test"])
    # labels = datasets["test"].get_labels()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    full_dataset = Messidor_Dataset("data/messidor", img_size=384)
    class_names = full_dataset.class_names
    datasets = {}
    datasets["test"] = full_dataset
    labels = datasets["test"].get_labels()

    # Save metrics for model
    model_directories = {
        "ViT-S-21k-384": r"runs\384_Strong_Aug\vit_small_patch16_224_in21k\LR_0.01",
        "ResNet50-21k-384": r"runs\384_Strong_Aug\resnetv2_50x1_bitm_in21k\LR_0.01",
        "ViT-S-DINO-384": r"runs\dino\vits_16\384\LR0.01",
        "ResNet50-DINO-384": r"runs\dino\resnet50\384\LR0.01",
        "ViT-S-21k-224": r"runs\224_Strong_Aug\vit_small_patch16_224_in21k\LR_0.01",
        "ResNet50-21k-224": r"runs\224_Strong_Aug\resnetv2_50x1_bitm_in21k\LR_0.01",
        "ViT-S-DINO-224": r"runs\dino\vits_16\224\LR0.01",
        "ResNet50-DINO-224": r"runs\dino\resnet50\224\LR0.01",
        }

    # model_dir = model_directories["ViT-S-21k-384"]
    # model = load_model("vit_small_patch16_224_in21k", device, class_names, model_dir, 384)
    # evaluate_model(model, device, model_dir, datasets, "test")

    # model_dir = model_directories["ResNet50-21k-384"]
    # model = load_model("resnetv2_50x1_bitm_in21k", device, class_names, model_dir)
    # evaluate_model(model, device, model_dir, datasets, "test")

    # model_dir = model_directories["ViT-S-DINO-384"]
    # model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')    
    # model = torch.nn.Sequential(model, torch.nn.Linear(384, 2))
    # model.load_state_dict(torch.load(os.path.join(model_dir, "model_params.pt")))
    # model = model.to(device)
    # evaluate_model(model, device, model_dir, datasets, "test")

    # model_dir = model_directories["ResNet50-DINO-384"]
    # model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')    
    # model = torch.nn.Sequential(model, torch.nn.Linear(2048, 2))
    # model.load_state_dict(torch.load(os.path.join(model_dir, "model_params.pt")))
    # model = model.to(device)
    # evaluate_model(model, device, model_dir, datasets, "test")

    # full_dataset = Messidor_Dataset("data/messidor", img_size=224)
    # class_names = full_dataset.class_names
    # datasets = {}
    # datasets["test"] = full_dataset
    # labels = datasets["test"].get_labels()

    # model_dir = model_directories["ViT-S-21k-224"]
    # model = load_model("vit_small_patch16_224_in21k", device, class_names, model_dir)
    # evaluate_model(model, device, model_dir, datasets, "test")

    # model_dir = model_directories["ResNet50-21k-224"]
    # model = load_model("resnetv2_50x1_bitm_in21k", device, class_names, model_dir)
    # evaluate_model(model, device, model_dir, datasets, "test")

    # model_dir = model_directories["ViT-S-DINO-224"]
    # model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')    
    # model = torch.nn.Sequential(model, torch.nn.Linear(384, 2))
    # model.load_state_dict(torch.load(os.path.join(model_dir, "model_params.pt")))
    # model = model.to(device)
    # evaluate_model(model, device, model_dir, datasets, "test")

    # model_dir = model_directories["ResNet50-DINO-224"]
    # model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')    
    # model = torch.nn.Sequential(model, torch.nn.Linear(2048, 2))
    # model.load_state_dict(torch.load(os.path.join(model_dir, "model_params.pt")))
    # model = model.to(device)
    # evaluate_model(model, device, model_dir, datasets, "test")

    # model_dir_ViT = r'runs\224_Strong_Aug\resnetv2_50x1_bitm_in21k\LR_0.01'
    # ViT = load_model("resnetv2_50x1_bitm_in21k", device, class_names, model_dir_ViT)
    # ViT = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')    
    # ViT = torch.nn.Sequential(ViT, torch.nn.Linear(2048, 2))
    # ViT.load_state_dict(torch.load(r'runs\dino\resnet50\224\LR0.01\model_params.pt'))
    # ViT = ViT.to(device)

    # evaluate_model(ViT, device, model_dir_ViT, datasets, "val")
    # evaluate_model(ViT, device, model_dir_ViT, datasets, "test")
    # exit()
    # BiT = load_model("resnetv2_50x1_bitm_in21k", device, class_names, model_dir_BiT)
    # evaluate_model(BiT, device, model_dir_BiT, datasets, "val")
    # evaluate_model(BiT, device, model_dir_BiT, datasets, "test")

    # Load metrics from model
    # metrics_ViT = load_metrics(model_dir_ViT, "test")
    # metrics_BiT = load_metrics(model_dir_BiT, "test")

    # Ensemble model
    # evaluate_ViT_BiT_ensemble_model(model_dir_ViT, model_dir_BiT, ensemble_dir, datasets) 
    # metrics_ensemble = load_metrics(ensemble_dir, "test")
    
    # Plot confusion matrices
    # fig, axes = plt.subplots(1, 3)
    # plot_confusion_matrix(labels, metrics_BiT["pred_log"], axes[0], "BiT prediction", "eyePACs label")
    # plot_confusion_matrix(labels, metrics_ViT["pred_log"], axes[1], "ViT prediction", "eyePACs label")
    # plot_confusion_matrix(labels, metrics_ensemble["pred_log"], axes[2], "Ensemble prediction", "eyePACs label")
    # axes[0].set_box_aspect(1)
    # axes[1].set_box_aspect(1)
    # plt.show()

    # Model comparision
    # Compare ViT and BiT predictions to ground truth
    # inter_model_matrix_comparision(labels, metrics_ViT["pred_log"], metrics_BiT["pred_log"], "Labels", ["Both Correct","Only ViT Correct","Only BiT Correct","Both Wrong"])
    # Quantiy which model ensemble predictions 'came from'
    # inter_model_matrix_comparision(metrics_ensemble["pred_log"], metrics_ViT["pred_log"], metrics_BiT["pred_log"], "Ensemble Prediction", ["ViT+BiT Agree","Choose ViT","Choose BiT","Choose Opposite"])

    # dataset_proportions = np.array([0.6, 0.2, 0.2])
    # full_dataset = EyePACS_Dataset(data_directory, random_state=13, img_size=384, labels_to_binary=False)
    # class_names = full_dataset.class_names
    # datasets = full_dataset.create_train_val_test_datasets(dataset_proportions, ["train", "val", "test"])
    # labels = datasets["test"].get_labels()
    # eval_dir = r"C:\Users\rmhisje\Documents\medical_ViT\eval_data"
    # image_dir = r"C:\Users\rmhisje\Documents\medical_ViT\diabetic-retinopathy-detection\preprocessed_448"
    # generate_folders_of_disagreements(eval_dir, image_dir, labels, metrics_ViT["pred_log"], metrics_BiT["pred_log"], datasets["train"].labels_df)

    # Kappa agreement
    from sklearn.metrics import cohen_kappa_score 
    # print(cohen_kappa_score(metrics_ViT["pred_log"], metrics_BiT["pred_log"]))

    # metrics_df = pd.DataFrame()
    # for name, directory in model_directories.items():
    #     metrics = load_metrics(directory, "test")
    #     for key in ["prob_log", "pred_log", "conf_matrix", "false_positive_rate", "threshold", "accuracy"]:
    #         del metrics[key]
        
    #     row = pd.DataFrame([list(metrics.values())], index=[name], columns=list(metrics.keys()))
    #     metrics_df = metrics_df.append(row)
    # metrics = [load_metrics(directory, "messidor_test") for directory in model_directories.values()]
    # print(metrics[0]["pred_log"])
    # print(labels)
    # plot_AUC_curves(labels, metrics, list(model_directories.keys()))

    data_eff = {
        "ViT-S-DINO": [0.645, 0.731, 0.757, 0.766],
        "ViT-S-21k": [0.683, 0.709, 0.759, 0.760],
        "ResNet50-DINO": [0.685, 0.707, 0.711, 0.710],
        "ResNet50-21k": [0.710, 0.776,  0.778, 0.807],
    }
    fractions = np.array([0.25, 0.5, 0.75, 1])*28
    for model, eff in data_eff.items():
        plt.plot(fractions, eff, label=model)
    plt.xlabel("Fraction of training data")
    plt.ylabel("Pre/Rec AUC")
    plt.legend()
    plt.show()