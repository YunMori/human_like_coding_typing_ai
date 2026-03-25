class TypingAIError(Exception):
    pass

class CodeGenerationError(TypingAIError):
    pass

class ASTParseError(TypingAIError):
    pass

class InjectionError(TypingAIError):
    pass

class ModelNotTrainedError(TypingAIError):
    pass

class UnsupportedLanguageError(TypingAIError):
    pass
