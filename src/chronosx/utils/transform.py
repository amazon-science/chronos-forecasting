import numpy as np
from gluonts.dataset.common import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    Chain,
    MapTransformation,
    SelectFields,
    Transformation,
    VstackFeatures,
)


class DropValues(MapTransformation):
    def __init__(
        self,
        drop_prob: float = 0.0,
        drop_mode: str = "upto",
        target_field: str = "target",
    ):
        assert drop_mode in ["upto", "exact"]
        assert 0.0 <= drop_prob <= 1.0

        self.drop_prob = drop_prob
        self.drop_mode = drop_mode
        self.target_field = target_field

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        if not is_train:
            return data

        target = data[self.target_field].copy().astype(float)

        if self.drop_mode == "exact":
            drop_p = self.drop_prob
        else:
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)

        mask = np.random.choice([True, False], size=len(target), p=[drop_p, 1 - drop_p])
        target[mask] = np.nan
        data[self.target_field] = target

        return data


def create_transformation(include_covariates: bool) -> Transformation:

    fields = [
        FieldName.START,
        FieldName.TARGET,
    ]
    if include_covariates:
        fields.append(FieldName.FEAT_DYNAMIC_REAL)

    transformation = (
        SelectFields(
            fields,
            allow_missing=True,
        )
        + DropValues(drop_prob=0.2, target_field=FieldName.TARGET)
        + AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        )
    )

    if include_covariates:
        field_time = (
            FieldName.FEAT_TIME
            if not include_covariates
            else FieldName.FEAT_DYNAMIC_REAL
        )
        assert field_time == "feat_dynamic_real"
        transformation = Chain(
            [
                transformation,
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_REAL, input_fields=[field_time]
                ),
            ]
        )

    return transformation
