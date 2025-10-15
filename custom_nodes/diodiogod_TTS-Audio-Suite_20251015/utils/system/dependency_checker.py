"""
Dependency checker for TTS Audio Suite initialization.
Warns users about missing dependencies based on enabled engines.
"""

import importlib
import sys
from typing import List, Dict, Tuple


class DependencyChecker:
    """Check for missing dependencies and provide helpful warnings."""
    
    # Core dependencies that should always be available
    CORE_DEPENDENCIES = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('soundfile', 'soundfile'),
        ('librosa', 'librosa'),
    ]
    
    # Engine-specific dependencies
    ENGINE_DEPENDENCIES = {
        'chatterbox': [
            ('s3tokenizer', 's3tokenizer'),
            ('transformers', 'transformers'),
            ('diffusers', 'diffusers'),
        ],
        'f5tts': [
            ('f5_tts', 'f5-tts'),
            ('cached_path', 'cached-path'),
        ],
        'higgs_audio': [
            ('dac', 'descript-audio-codec'),
            ('vector_quantize_pytorch', 'vector-quantize-pytorch'),
            ('dacite', 'dacite'),
        ],
        'rvc': [
            ('faiss', 'faiss-cpu'),
            ('onnxruntime', 'onnxruntime-gpu'),
            ('torchcrepe', 'torchcrepe'),
        ]
    }
    
    @staticmethod
    def check_dependency(module_name: str) -> bool:
        """Check if a dependency is available."""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
        except (AttributeError, ValueError) as e:
            # Handle problematic packages like faiss with circular import issues
            if 'faiss' in module_name and ('circular import' in str(e) or 'Float32' in str(e)):
                # faiss has circular import issues in some environments (Docker, etc.)
                # If we get AttributeError about Float32, faiss is installed but has import issues
                return True
            return False
    
    @staticmethod
    def check_core_dependencies() -> List[Tuple[str, str]]:
        """Check core dependencies that are always needed."""
        missing = []
        for module_name, package_name in DependencyChecker.CORE_DEPENDENCIES:
            if not DependencyChecker.check_dependency(module_name):
                missing.append((module_name, package_name))
        return missing
    
    @staticmethod
    def check_engine_dependencies(engine: str) -> List[Tuple[str, str]]:
        """Check dependencies for a specific engine."""
        if engine not in DependencyChecker.ENGINE_DEPENDENCIES:
            return []
        
        missing = []
        for module_name, package_name in DependencyChecker.ENGINE_DEPENDENCIES[engine]:
            if not DependencyChecker.check_dependency(module_name):
                missing.append((module_name, package_name))
        return missing
    
    @staticmethod
    def get_missing_dependencies_report() -> str:
        """Generate a comprehensive missing dependencies report."""
        report_lines = []
        
        # Check core dependencies
        core_missing = DependencyChecker.check_core_dependencies()
        if core_missing:
            report_lines.append("⚠️  CRITICAL: Missing core dependencies:")
            for module_name, package_name in core_missing:
                report_lines.append(f"   • {package_name} (import: {module_name})")
            report_lines.append("")
        
        # Check engine-specific dependencies
        engine_issues = {}
        for engine in DependencyChecker.ENGINE_DEPENDENCIES:
            missing = DependencyChecker.check_engine_dependencies(engine)
            if missing:
                engine_issues[engine] = missing
        
        if engine_issues:
            report_lines.append("⚠️  Engine-specific missing dependencies:")
            for engine, missing_deps in engine_issues.items():
                engine_display = {
                    'chatterbox': 'ChatterBox TTS',
                    'f5tts': 'F5-TTS',
                    'higgs_audio': 'Higgs Audio 2',
                    'rvc': 'RVC Voice Conversion'
                }.get(engine, engine)
                
                report_lines.append(f"   {engine_display}:")
                for module_name, package_name in missing_deps:
                    report_lines.append(f"     • {package_name} (import: {module_name})")
            report_lines.append("")
        
        if core_missing or engine_issues:
            report_lines.append("🔧 To fix: pip install -r requirements.txt")
            report_lines.append("   Or install specific packages: pip install <package_name>")
            
            if engine_issues:
                report_lines.append("")
                report_lines.append("ℹ️  Note: Engine nodes will fail to load without their dependencies")
        
        return "\n".join(report_lines) if report_lines else ""
    
    @staticmethod
    def get_startup_warnings() -> List[str]:
        """Get dependency warnings in ComfyUI startup format (list of strings)."""
        warnings = []
        
        # Check core dependencies
        core_missing = DependencyChecker.check_core_dependencies()
        if core_missing:
            warnings.append("⚠️ Critical dependencies missing:")
            for module_name, package_name in core_missing:
                warnings.append(f"   • {package_name} (import: {module_name})")
        
        # Check engine-specific dependencies  
        engine_issues = {}
        for engine in DependencyChecker.ENGINE_DEPENDENCIES:
            missing = DependencyChecker.check_engine_dependencies(engine)
            if missing:
                engine_issues[engine] = missing
        
        if engine_issues:
            warnings.append("⚠️ Engine dependencies missing:")
            for engine, missing_deps in engine_issues.items():
                engine_display = {
                    'chatterbox': 'ChatterBox TTS',
                    'f5tts': 'F5-TTS', 
                    'higgs_audio': 'Higgs Audio 2',
                    'rvc': 'RVC Voice Conversion'
                }.get(engine, engine)
                
                warnings.append(f"   {engine_display}:")
                for module_name, package_name in missing_deps:
                    warnings.append(f"     • {package_name} (import: {module_name})")
        
        if core_missing or engine_issues:
            warnings.append("🔧 Fix: pip install -r requirements.txt")
            warnings.append("ℹ️ Engine nodes will fail without dependencies")
        
        return warnings