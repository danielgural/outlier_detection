import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone.brain import Similarity
from pprint import pprint
from fiftyone import ViewField as F


import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone.core.utils as fou
import numpy as np
import fiftyone.zoo as foz

sklearn = fou.lazy_import("sklearn")
umap = fou.lazy_import("umap")




class OutlierDetection(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="outlier_detection",
            label="Outlier Detection",
            description="Finds outliers in your dataset based off a customizale threshold",
            icon="/assets/binoculars.svg",
            dynamic=True,

        )
    
    def resolve_input(self, ctx):
        inputs = types.Object()
    
        ready = _outlier_detection_inputs(ctx,inputs)

        if ready:
            _execution_mode(ctx, inputs)
        

        return types.Property(inputs)
    
    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)
    
    def execute(self, ctx):
        
        _outlier_detections(ctx)
    
        return {}
    
def _outlier_detection_inputs(ctx, inputs):

    target_view = get_target_view(ctx, inputs)

    model_choices = ["alexnet-imagenet-torch",
                     "classification-transformer-torch",
                     "clip-vit-base32-torch",
                     "densenet121-imagenet-torch",
                     "densenet161-imagenet-torch",
                     "densenet169-imagenet-torch",
                     "densenet201-imagenet-torch",
                     "detection-transformer-torch",
                     "dinov2-vitb14-torch",
                     "dinov2-vitg14-torch",
                     "dinov2-vitl14-torch",
                     "dinov2-vits14-torch",
                     "googlenet-imagenet-torch",
                     "inception-resnet-v2-imagenet-tf1",
                     "inception-v3-imagenet-torch",
                     "inception-v4-imagenet-tf1",
                     "mobilenet-v2-imagenet-tf1",
                     "mobilenet-v2-imagenet-torch",
                     "open-clip-torch",
                     "resnet-v1-50-imagenet-tf1",
                     "resnet-v2-50-imagenet-tf1",
                     "resnet101-imagenet-torch",
                     "resnet152-imagenet-torch",
                     "resnet18-imagenet-torch",
                     "resnet34-imagenet-torch",
                     "resnet50-imagenet-torch",
                     "resnext101-32x8d-imagenet-torch",
                     "resnext50-32x4d-imagenet-torch",
                     "vgg11-bn-imagenet-torch",
                     "vgg11-imagenet-torch",
                     "vgg13-bn-imagenet-torch",
                     "vgg13-imagenet-torch",
                     "vgg16-bn-imagenet-torch",
                     "vgg16-imagenet-tf1",
                     "vgg16-imagenet-torch",
                     "vgg19-bn-imagenet-torch",
                     "vgg19-imagenet-torch",
                     "wide-resnet101-2-imagenet-torch",
                     "wide-resnet50-2-imagenet-torch",
                     "zero-shot-classification-transformer-torch",
                     "zero-shot-detection-transformer-torch"
                      ] 

    model_radio_group = types.RadioGroup()

    for choice in model_choices:
        model_radio_group.add_choice(choice, label=choice)

    inputs.enum(
        "model_radio_group",
        model_radio_group.values(),
        label="Embedding model to use",
        description="Choose what model will generate your embeddings:",
        view=types.DropdownView(),
        default='clip-vit-base32-torch',
        required=False,
        )
    
    inputs.bool(
            "force_embeddings",
            label="Force generate new embeddings?",
            description="This will overwrite any previous embeddings",
            view=types.SwitchView(),
            default=False
            )


    outlier_alg_choices = ["Local Outlier Factor", ] 


    outlier_radio_group = types.RadioGroup()

    for choice in outlier_alg_choices:
        outlier_radio_group.add_choice(choice, label=choice)

    inputs.enum(
    "outlier_radio_group",
    outlier_radio_group.values(),
    label="Outlier Detection Algorithm",
    description="Choose what algorithm to use:",
    view=types.DropdownView(),
    default='Local Outlier Factor',
    required=False,
    )

    outlier_alg = ctx.params.get("outlier_radio_group", False)

    if outlier_alg == "Local Outlier Factor":

        inputs.float(
            "contamination",
            label="Contamination",
            description="Percentage of samples you want to find outliers from (bounded (0,1))",
            view=types.FieldView(componentsProps={'field': {'min': 0.0001, "max": 0.9999, "step": 0.01, "default": 0.01}}),
            )

    inputs.bool(
            "tag_samples",
            label="Tag Outliers?",
            description="Turn on to tag outliers found",
            view=types.SwitchView(),
            )

    tag_samples = ctx.params.get("tag_samples",False)

    if tag_samples:
        inputs.str(
            "tag_name",
            label="What would like the tag to be?",
            description="Name the tag for your outliers",
            default="Outliers"
            )


    inputs.bool(
        "filter_by_class",
        label="Filter by class?",
        description="Turn on filtering on a specific class or not.",
        view=types.SwitchView(),
        )

    by_class = ctx.params.get("filter_by_class", False)

    if by_class:
        labels = []
        field_names = list(target_view.get_field_schema().keys())
        for name in field_names:
            if type(target_view.get_field(name)) == fo.core.fields.EmbeddedDocumentField:
                if "detections" in  list(target_view.get_field(name).get_field_schema().keys()):
                    labels.append(name + ".detections")
                elif "label" in list(target_view.get_field(name).get_field_schema().keys()):
                    labels.append(name)

        if labels == []:
            inputs.view(
            "error", 
            types.Error(label="No labels found on this dataset", description="Add labels to be able to filter by them")
        )
        else:

            label_radio_group = types.RadioGroup()

            for choice in labels:
                label_radio_group.add_choice(choice, label=choice)

            inputs.enum(
                "label_radio_group",
                label_radio_group.values(),
                label="Choose Field",
                description="Choose what label field to filter on:",
                view=types.DropdownView(),
                required=True,
                default=None
                )


            field = ctx.params.get("label_radio_group")
            if field == None:
                inputs.view(
                    "warning", 
                    types.Error(label="Choose a field first!", description="Pick a label field to filter on first")
                )
            else:
                classes = target_view.distinct(field + ".label")
                class_radio_group = types.RadioGroup()

                for choice in classes:
                    class_radio_group.add_choice(choice, label=choice)

                inputs.enum(
                "class_radio_group",
                class_radio_group.values(),
                label="Choose Class",
                description="Choose what class to filter on:",
                view=types.DropdownView(),
                required=True,
                )

    return True



def _outlier_detections(ctx):
    outlier_alg = ctx.params.get("outlier_radio_group")
    field = ctx.params.get("label_radio_group")
    by_class = ctx.params.get("filter_by_class")
    target = ctx.params.get("target", None)
    target_view = _get_target_view(ctx, target)
    tag_samples = ctx.params.get("tag_samples")
    tag_name = ctx.params.get("tag_name", None)
    model_choice = ctx.params.get("model_radio_group")
    force_embeddings = ctx.params.get("force_embeddings")

    model = foz.load_zoo_model(model_choice)

    if outlier_alg == "Local Outlier Factor":
        contamination = ctx.params.get("contamination")
        if by_class:
            cls = ctx.params.get("class_radio_group")
            target_view = target_view.filter_labels(
                field, (F("label") == cls)
            )
            
        if "embeddings" not in list(target_view.get_field_schema().keys()) or force_embeddings:
            target_view.compute_embeddings(model, embeddings_field="embeddings")


        embeddings = np.array(target_view.values("embeddings"))
        mapper = umap.UMAP().fit(embeddings)
        outlier_scores = sklearn.neighbors.LocalOutlierFactor(contamination=contamination).fit_predict(mapper.embedding_)
        outliers = target_view[outlier_scores == -1]

        if tag_samples:
            outliers.tag_samples(tag_name)

        ctx.trigger("set_view", {"view": outliers._serialize()})




    return



def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )



def get_target_view(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = None

    if has_view or has_selected:
        target_choices = types.RadioGroup(orientation="horizontal")
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Process the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Process the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Process only the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            required=True,
            label="Target view",
            view=target_choices,
        )

    target = ctx.params.get("target", default_target)

    return _get_target_view(ctx, target)

def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view

def register(plugin):
    plugin.register(OutlierDetection)
