plugins {
	id("com.android.application")
	id("org.jetbrains.kotlin.android")
	id("org.jetbrains.kotlin.plugin.compose")
}

android {
	namespace = "com.klstudio.rubikdetector"
	compileSdk = 34

	defaultConfig {
		applicationId = "com.klstudio.rubikdetector"
		minSdk = 26
		targetSdk = 34
		versionCode = 1
		versionName = "0.1"
		ndk {
			abiFilters += listOf("arm64-v8a", "x86_64")
		}
		externalNativeBuild {
			cmake {
				cppFlags += listOf("-std=c++17", "-fexceptions", "-frtti")
				arguments += listOf("-DANDROID_STL=c++_shared")
			}
		}
	}

	buildTypes {
		release {
			isMinifyEnabled = false
			signingConfig = signingConfigs.getByName("debug")
		}
	}

	externalNativeBuild {
		cmake {
			path = file("src/main/cpp/CMakeLists.txt")
			version = "3.22.1"
		}
	}

	compileOptions {
		sourceCompatibility = JavaVersion.VERSION_17
		targetCompatibility = JavaVersion.VERSION_17
	}
	kotlinOptions {
		jvmTarget = "17"
	}
	buildFeatures {
		compose = true
	}
}

dependencies {
	val camerax = "1.3.4"
	implementation("androidx.camera:camera-core:$camerax")
	implementation("androidx.camera:camera-camera2:$camerax")
	implementation("androidx.camera:camera-lifecycle:$camerax")
	implementation("androidx.camera:camera-view:$camerax")

	implementation(platform("androidx.compose:compose-bom:2024.09.02"))
	implementation("androidx.compose.ui:ui")
	implementation("androidx.compose.foundation:foundation")
	implementation("androidx.compose.material3:material3")
	implementation("androidx.activity:activity-compose:1.9.2")
	implementation("androidx.core:core-ktx:1.13.1")
	implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
}
