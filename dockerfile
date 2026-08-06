# Stage 1: Build stage
FROM eclipse-temurin:21-jdk AS builder
WORKDIR /application

# 1. Copy build configuration files
COPY gradlew .
COPY gradle gradle
COPY build.gradle* .
COPY settings.gradle* .
COPY *.properties .

# 2. Fix Windows CRLF line endings in gradlew wrapper script
RUN cat gradlew | tr -d '\r' > gradlew.linux && \
    mv gradlew.linux gradlew && \
    chmod +x gradlew

# 3. Copy source code
COPY src src

# 4. Compile fat JAR using Gradle caching
RUN --mount=type=cache,target=/root/.m2 \
    --mount=type=cache,target=/root/.gradle \
    ./gradlew clean bootJar --no-daemon --no-configuration-cache

# 5. Extract layers for Spring Boot optimize extraction
RUN cp build/libs/*.jar app.jar && \
    java -Djarmode=layertools -jar app.jar extract

# ==========================================
# Stage 2: Final Runtime Image (Ubuntu/Debian)
# ==========================================
FROM eclipse-temurin:21-jre
LABEL author="Gerrits Marc"

# Security: Non-root user setup for Debian/Ubuntu (using groupadd/useradd)
RUN groupadd -r spring && useradd -r -g spring spring
USER spring:spring
WORKDIR /application

# Copy extracted layers from builder
COPY --from=builder /application/dependencies/ ./
COPY --from=builder /application/spring-boot-loader/ ./
COPY --from=builder /application/snapshot-dependencies/ ./
COPY --from=builder /application/application/ ./

ENV JAVA_OPTS="-Xms256m -Xmx384m \
               -XX:+UseG1GC \
               -XX:TieredStopAtLevel=1 \
               -Dspring.threads.virtual.enabled=true \
               -Dspring.main.lazy-initialization=true \
               -Dspring.data.jpa.repositories.bootstrap-mode=deferred"

ENV SPRING_PROFILES_ACTIVE=render-service

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS org.springframework.boot.loader.launch.JarLauncher"]