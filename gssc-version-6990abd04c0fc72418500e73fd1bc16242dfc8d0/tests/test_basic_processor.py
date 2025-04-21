import pytest
import numpy as np
from gssc.basic_processor import BasicProcessor

def test_channel_validation():
    """Test channel validation against required configuration"""
    valid_channels = ["C3", "C4", "F3", "F4", "HEOG"]
    invalid_channels = ["Cz", "P3", "HEOG"]  # Missing required EEG channels
    
    # Test valid channel configuration
    bp_valid = BasicProcessor(channels=valid_channels)
    assert bp_valid.validate_channels(), "Should validate correct channels"
    
    # Test invalid configuration
    with pytest.raises(ValueError):
        BasicProcessor(channels=invalid_channels)

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline"""
    test_data = np.random.randn(5, 1000)  # 5 channels, 1000 samples
    bp = BasicProcessor()
    
    processed = bp.preprocess(test_data)
    
    assert processed.shape == test_data.shape, "Output shape should match input"
    assert not np.allclose(processed, test_data), "Data should be modified by preprocessing"

def test_feature_extraction():
    """Test feature extraction outputs"""
    bp = BasicProcessor()
    test_window = np.random.randn(5, 256)  # 5 channels, 1-second window @256Hz
    
    features = bp.extract_features(test_window)
    
    assert isinstance(features, dict), "Features should be dictionary"
    assert "bandpower" in features, "Should contain bandpower features"
    assert "hjorth" in features, "Should contain Hjorth parameters"

def test_eog_artifact_handling():
    """Test EOG artifact detection and handling"""
    # Create synthetic data with artifact
    clean_data = np.random.randn(5, 1000) * 1e-6  # 5 channels
    artifact_data = clean_data.copy()
    artifact_data[-1] += 1.0  # Add large EOG artifact
    
    bp = BasicProcessor()
    
    # Test artifact detection
    assert bp.detect_artifacts(artifact_data), "Should detect EOG artifact"
    
    # Test artifact correction
    corrected = bp.correct_artifacts(artifact_data)
    assert np.allclose(corrected[-1], clean_data[-1], atol=1e-3), "Should correct EOG channel"

def test_real_time_handling():
    """Test streaming data handling capabilities"""
    bp = BasicProcessor()
    chunk_size = 256  # 1 second @256Hz
    
    # Simulate streaming input
    for _ in range(10):
        chunk = np.random.randn(5, chunk_size)
        bp.process_chunk(chunk)
    
    assert bp.buffer.shape == (5, 10*chunk_size), "Should maintain proper buffer"

def test_model_integration():
    """Test integration with inference model"""
    bp = BasicProcessor()
    test_window = np.random.randn(5, 256)
    
    # Mock model
    class MockModel:
        def predict(self, features):
            return 0.8
    
    bp.model = MockModel()
    prediction = bp.run_inference(test_window)
    
    assert 0 <= prediction <= 1, "Should return valid probability"

def test_edge_cases():
    """Test handling of edge cases and invalid inputs"""
    bp = BasicProcessor()
    
    # Test empty input
    with pytest.raises(ValueError):
        bp.process_chunk(np.array([]))
    
    # Test incorrect sampling rate
    with pytest.raises(ValueError):
        BasicProcessor(sampling_rate=100)  # Unsupported rate
    
    # Test invalid data dimensions
    with pytest.raises(ValueError):
        bp.process_chunk(np.random.randn(3, 100))  # Incorrect channel count 