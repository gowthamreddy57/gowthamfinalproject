# Copyright 2022 The mT5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Import API modules."""

# Core mT5 modules (with Windows compatibility via stubs)
try:
    import multilingual_t5.preprocessors
    import multilingual_t5.tasks
    import multilingual_t5.utils
    MT5_CORE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Core mT5 modules not fully available: {e}")
    MT5_CORE_AVAILABLE = False

# Grok API Integration (optional - requires GROK_API_KEY)
try:
    from multilingual_t5.grok_client import (
        GrokClient,
        GrokConfig,
        GrokModel,
        TaskType,
        create_client,
        quick_translate,
        quick_qa,
    )
    from multilingual_t5.grok_evaluator import (
        EvaluationRunner,
        EvaluationConfig,
        EvaluationReport,
    )
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

