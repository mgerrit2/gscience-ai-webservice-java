package com.gscience.ai.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record PhotoTrainRecord(
        @JsonProperty("photo_id") String photoId,
        @JsonProperty("business_id") String businessId,
        String caption,
        String label
) {}