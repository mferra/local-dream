package io.github.xororz.localdream

import android.app.Application
import coil.ImageLoader
import coil.ImageLoaderFactory
import io.github.xororz.localdream.data.HistoryMigration
import io.github.xororz.localdream.data.MigrationState
import io.github.xororz.localdream.data.db.AppDatabase
import io.github.xororz.localdream.utils.Http
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class LocalDreamApplication :
    Application(),
    ImageLoaderFactory {

    private val appScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val _migrationState = MutableStateFlow<MigrationState>(MigrationState.Idle)
    val migrationState: StateFlow<MigrationState> = _migrationState.asStateFlow()

    private var migrationJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        startMigration()
    }

    // Coil would otherwise build its own OkHttp client; share the app's so image
    // loading is held to the same outbound allowlist.
    override fun newImageLoader(): ImageLoader = ImageLoader.Builder(this)
        .okHttpClient { Http.client }
        .build()

    private fun startMigration() {
        migrationJob?.cancel()
        migrationJob = appScope.launch {
            try {
                if (HistoryMigration.isDone(this@LocalDreamApplication)) {
                    _migrationState.value = MigrationState.NotNeeded
                    return@launch
                }
                HistoryMigration.migrate(
                    this@LocalDreamApplication,
                    AppDatabase.get(this@LocalDreamApplication),
                    _migrationState,
                )
            } catch (e: Throwable) {
                _migrationState.value = MigrationState.Failed(e)
            }
        }
    }

    fun retryMigration() {
        _migrationState.value = MigrationState.Idle
        startMigration()
    }

    fun skipMigration() {
        migrationJob?.cancel()
        appScope.launch {
            try {
                HistoryMigration.markDoneExternal(this@LocalDreamApplication)
            } catch (_: Throwable) {
                // ignore — UI will still proceed
            }
            _migrationState.value = MigrationState.Done
        }
    }
}
